import tempfile
import unittest
from pathlib import Path

from frametag.domain import (
    filename_from_analysis, list_images, natural_key, normalize_analysis,
    sanitize_filename, select_through, unify_runs,
)
from frametag.models import Photo


def photo(number, gender="남성", objects=None, **flags):
    return Photo(Path(f"/photos/frame{number}.jpg"), 100, 200,
                 {"gender": gender, "clothing": ["검정반팔", "파랑청바지"],
                  "objects": objects or []}, status="완료", **flags)


class AnalysisTests(unittest.TestCase):
    def test_allowed_clothes_outer_priority_and_blue(self):
        result = normalize_analysis({
            "gender": "male", "clothing": [
                {"color": "black", "type": "t-shirt"},
                {"color": "카키", "type": "조끼"},
                {"color": "navy", "type": "jeans"},
                {"color": "흰색", "type": "긴바지"},
                {"color": "회색", "type": "양복"},
                {"color": "없는색", "type": "우비"},
            ],
        })
        self.assertEqual(result, {"gender": "남성", "clothing": ["카키조끼", "파랑청바지"], "objects": []})

    def test_dress_and_no_person(self):
        raw = {"gender": "여성", "clothing": ["빨강원피스", "파랑청바지", "검정반팔"]}
        self.assertEqual(normalize_analysis(raw)["clothing"], ["빨강원피스"])
        raw["gender"] = "인물없음"
        raw["objects"] = [{"name": "가방", "location": "바닥"}]
        self.assertEqual(normalize_analysis(raw), {"gender": "인물없음", "clothing": [], "objects": []})

    def test_construction_details(self):
        result = normalize_analysis({"shirt": {"type": "반팔", "color": "흰색", "has_collar": True},
                                     "pants": {"type": "긴바지", "color": "진청", "material": "denim"}})
        self.assertEqual(result["clothing"], ["흰색셔츠", "파랑청바지"])

    def test_floor_exclusions_and_bottles(self):
        raw = {"gender": "불명", "objects": [
            {"name": name, "location": "바닥"} for name in ("의자", "종이", "신문", "매트", "신발", "휴대폰")
        ] + [{"name": "휴대폰", "location": "손"}, {"name": "생수병", "location": "floor"},
             {"name": "glass bottle", "location": "hand"}, {"name": "우산", "location": "unknown"}]}
        self.assertEqual(normalize_analysis(raw)["objects"], ["병(손)", "휴대폰(손)", "페트병(바닥)"])


class FilenameTests(unittest.TestCase):
    def test_generated_and_manual_object_rules(self):
        analysis = {"gender": "남성", "clothing": ["검정반팔", "파랑청바지"],
                    "objects": ["가방(손)", "캐리어(바닥)"]}
        self.assertEqual(filename_from_analysis(analysis, ".jpg"),
                         "남성_검정반팔_파랑청바지_손-가방_바닥-캐리어.jpg")
        analysis.update(clothing=["내가 쓴 옷"], objects=["의자(바닥)", "종이", "우산"])
        self.assertEqual(filename_from_analysis(analysis, ".JPG"), "남성_내가쓴옷_바닥-의자_종이_우산.JPG")

    def test_manual_names_never_escape_folder_or_change_format(self):
        for name in ("../outside", r"..\outside", "C:outside", "AUX", "con.txt", "CON .txt", "CONIN$", "LPT¹", "...", "x ", "x.", "x\x00", "x.png"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                sanitize_filename(name, ".jpg")
        self.assertEqual(sanitize_filename("새 사진", ".jpg"), "새 사진.jpg")
        self.assertEqual(sanitize_filename("새 사진.JPG", ".jpg"), "새 사진.jpg")

    def test_generated_input_is_safe_and_long_korean_is_bounded(self):
        result = filename_from_analysis({"gender": "CON", "clothing": ["옷" * 500],
                                         "objects": ["../outside(손)"]}, ".png")
        self.assertLessEqual(len(result.encode("utf-8")), 240)
        self.assertEqual(Path(result).suffix, ".png")
        self.assertNotIn("/", result)
        self.assertEqual(sanitize_filename(result, ".png"), result)


class SequenceTests(unittest.TestCase):
    def test_majority_objects_and_gender(self):
        photos = [photo(1, objects=["가방(손)"]), photo(2), photo(3, objects=["가방(손)"])]
        self.assertEqual(unify_runs(photos), [photos[1]])
        self.assertEqual(photos[1].analysis["objects"], ["가방(손)"])
        self.assertEqual(photos[1].revision, 1)
        self.assertEqual(photos[1].status, "완료")
        photos = [photo(1, "불명"), photo(2, "여성"), photo(3, "여성")]
        self.assertEqual(unify_runs(photos), [photos[0]])
        self.assertEqual(photos[0].analysis["gender"], "여성")

    def test_ties_prefer_more_objects_and_known_gender(self):
        photos = [photo(1), photo(2, objects=["가방(손)"]), photo(3, objects=["가방(손)", "우산(손)"])]
        unify_runs(photos)
        self.assertTrue(all(p.analysis["objects"] == ["가방(손)", "우산(손)"] for p in photos))
        photos = [photo(1, "불명"), photo(2, "남성"), photo(3, "여성")]
        unify_runs(photos)
        self.assertTrue(all(p.analysis["gender"] == "남성" for p in photos))

    def test_manual_confirmed_and_folder_boundaries(self):
        for flag in ("manual", "confirmed"):
            photos = [photo(1), photo(2, objects=["우산(손)"], **{flag: True}), photo(3)]
            self.assertEqual(unify_runs(photos), [])
            self.assertEqual(photos[1].analysis["objects"], ["우산(손)"])
        photos = [photo(1), photo(2), photo(3, objects=["우산(손)"])]
        photos[2].path = Path("/other/frame3.jpg")
        self.assertEqual(unify_runs(photos), [])

    def test_select_through_excludes_already_confirmed(self):
        photos = [photo(1, confirmed=True), photo(2), photo(3), photo(4)]
        self.assertEqual(select_through(photos, photos[2]), photos[1:3])
        self.assertEqual(select_through(photos, photos[0]), [])
        self.assertEqual(select_through(photos, photo(8)), [])


class ImageListTests(unittest.TestCase):
    def test_natural_sort_supported_formats_and_recursion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("photo10.JPG", "photo2.png", "photo1.jpg", "note.txt", "photo3.gif"):
                (root / name).touch()
            (root / "sub").mkdir()
            (root / "sub" / "photo1.webp").touch()
            self.assertEqual([p.name for p in list_images(root)], ["photo1.jpg", "photo2.png", "photo3.gif", "photo10.JPG"])
            self.assertEqual(len(list_images(root, recursive=True)), 5)
            self.assertLess(natural_key("pic2"), natural_key("pic10"))


if __name__ == "__main__":
    unittest.main()
