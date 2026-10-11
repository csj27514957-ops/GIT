"""Windows entry point. Run with --background to open in the system tray."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="FrameTag 사진 이름 정리")
    parser.add_argument("--background", action="store_true", help="트레이에서 시작")
    args = parser.parse_args()
    from frametag.ui import FrameTagApp
    app = FrameTagApp(background=args.background)
    app.mainloop()


if __name__ == "__main__":
    main()
