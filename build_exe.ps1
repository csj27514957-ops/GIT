# Build on Windows using Python 3.12 or newer. Run from the source directory.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @()
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

try {
    if ($env:OS -ne 'Windows_NT') {
        throw 'Build the Windows EXE on Windows. Cross-compilation is not supported.'
    }
    $projectPath = $PSScriptRoot
    $appPath = Join-Path $projectPath 'app.py'
    $requirementsPath = Join-Path $projectPath 'requirements-dev.txt'
    if (!(Test-Path -LiteralPath $appPath -PathType Leaf) -or
        !(Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
        throw 'Extract the complete source package before building.'
    }
    $venvPath = Join-Path $projectPath '.venv'
    $pythonPath = Join-Path $venvPath 'Scripts\python.exe'
    $versionCheck = 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'
    if (!(Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        $created = $false
        $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
        if ($null -ne $launcher) {
            & $launcher.Source -3 -c $versionCheck
            if ($LASTEXITCODE -eq 0) {
                Invoke-Checked -Executable $launcher.Source -Arguments @('-3', '-m', 'venv', $venvPath)
                $created = $true
            }
        }
        if (!$created) {
            $systemPython = Get-Command 'python.exe' -ErrorAction SilentlyContinue
            if ($null -eq $systemPython) {
                throw 'Install Python 3.12 or newer for Windows, including Tcl/Tk.'
            }
            Invoke-Checked -Executable $systemPython.Source -Arguments @('-c', $versionCheck)
            Invoke-Checked -Executable $systemPython.Source -Arguments @('-m', 'venv', $venvPath)
        }
    }
    Invoke-Checked -Executable $pythonPath -Arguments @('-c', 'import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 12) else 1)')
    Invoke-Checked -Executable $pythonPath -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', $requirementsPath)

    $distPath = Join-Path $projectPath 'dist'
    $buildPath = Join-Path $projectPath 'build'
    New-Item -ItemType Directory -Path $distPath -Force | Out-Null
    New-Item -ItemType Directory -Path $buildPath -Force | Out-Null
    Invoke-Checked -Executable $pythonPath -Arguments @(
        '-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile', '--windowed',
        '--name', 'FrameTag', '--hidden-import', 'keyring.backends.Windows',
        '--hidden-import', 'pystray._win32', '--distpath', $distPath,
        '--workpath', $buildPath, '--specpath', $buildPath, $appPath
    )
    $exePath = Join-Path $distPath 'FrameTag.exe'
    if (!(Test-Path -LiteralPath $exePath -PathType Leaf)) {
        throw 'PyInstaller did not produce FrameTag.exe.'
    }
    $hashPath = Join-Path $distPath 'FrameTag_SHA256.txt'
    $hash = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  FrameTag.exe" | Set-Content -LiteralPath $hashPath -Encoding Ascii

    $readmePath = Join-Path $distPath 'README.md'
    $manualName = 'FrameTag_' + [char]0xc0ac + [char]0xc6a9 + [char]0xc124 + [char]0xba85 + [char]0xc11c + '.txt'
    $manualPath = Join-Path $distPath $manualName
    Copy-Item -LiteralPath (Join-Path $projectPath 'README.md') -Destination $readmePath -Force
    Copy-Item -LiteralPath (Join-Path $projectPath $manualName) -Destination $manualPath -Force
    $archivePath = Join-Path $projectPath 'FrameTag-distribution.zip'
    Compress-Archive -LiteralPath @($exePath, $hashPath, $readmePath, $manualPath) -DestinationPath $archivePath -Force
    Write-Host "EXE: $exePath"
    Write-Host "ZIP: $archivePath"
    Write-Host 'This is an unsigned build. Keep the original source package separately.'
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
