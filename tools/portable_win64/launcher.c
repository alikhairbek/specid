/* SpecID Studio launcher.
   Starts runtime\pythonw.exe app\specid_studio.py from the folder of this executable (no console window).
   If the private runtime, the packages or the model assets are missing, it first runs setup\bootstrap.bat in a console
   (downloads them into this folder; Internet needed once), then starts the application. */
#include <windows.h>
#include <wchar.h>

#define N 4096
static void fail(const wchar_t *msg) { MessageBoxW(NULL, msg, L"SpecID Studio", MB_ICONERROR | MB_OK); }
static int exists(const wchar_t *p) { return GetFileAttributesW(p) != INVALID_FILE_ATTRIBUTES; }
static int ready(const wchar_t *dir) {
    wchar_t a[N], b[N], c[N];
    _snwprintf(a, N, L"%ls\\runtime\\pythonw.exe", dir); _snwprintf(b, N, L"%ls\\runtime\\specid_runtime_ok.txt", dir); _snwprintf(c, N, L"%ls\\app\\assets\\index.json", dir);
    return exists(a) && exists(b) && exists(c);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR args, int nShow) {
    wchar_t dir[N], py[N], script[N], appdir[N], bat[N], cmd[3 * N], msg[2 * N];
    DWORD n = GetModuleFileNameW(NULL, dir, N);
    if (n == 0 || n >= N) { fail(L"Cannot determine the program folder."); return 1; }
    wchar_t *slash = wcsrchr(dir, L'\\'); if (slash) *slash = 0;
    _snwprintf(script, N, L"%ls\\app\\specid_studio.py", dir); _snwprintf(appdir, N, L"%ls\\app", dir); _snwprintf(bat, N, L"%ls\\setup\\bootstrap.bat", dir);
    if (!exists(script)) {
        _snwprintf(msg, 2 * N, L"The application files were not found:\n%ls\n\nExtract the whole SpecID_Studio folder from the zip (do not run the program from inside the zip).", script);
        fail(msg); return 1;
    }
    if (!ready(dir)) {
        if (!exists(bat)) { _snwprintf(msg, 2 * N, L"The setup script was not found:\n%ls\n\nExtract the whole SpecID_Studio folder from the zip.", bat); fail(msg); return 1; }
        _snwprintf(cmd, 3 * N, L"cmd.exe /c \"\"%ls\"\"", bat);
        STARTUPINFOW si; PROCESS_INFORMATION pi; ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si); ZeroMemory(&pi, sizeof(pi));
        if (!CreateProcessW(NULL, cmd, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, dir, &si, &pi)) { _snwprintf(msg, 2 * N, L"Could not start the setup (error %lu):\n%ls", GetLastError(), bat); fail(msg); return 1; }
        WaitForSingleObject(pi.hProcess, INFINITE); DWORD code = 1; GetExitCodeProcess(pi.hProcess, &code); CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        if (code != 0 || !ready(dir)) {
            _snwprintf(msg, 2 * N, L"The first-run setup did not complete (code %lu).\n\nDetails: %ls\\setup\\setup_log.txt\n\nRun SpecID_Studio.exe again when the Internet connection is available: the setup continues where it stopped.", code, dir);
            fail(msg); return 1;
        }
    }
    _snwprintf(py, N, L"%ls\\runtime\\pythonw.exe", dir);
    SetEnvironmentVariableW(L"PYTHONHOME", NULL); SetEnvironmentVariableW(L"PYTHONPATH", NULL);
    _snwprintf(cmd, 3 * N, L"\"%ls\" -E -s -X utf8 \"%ls\" %ls", py, script, args ? args : L"");
    STARTUPINFOW si; PROCESS_INFORMATION pi; ZeroMemory(&si, sizeof(si)); si.cb = sizeof(si); ZeroMemory(&pi, sizeof(pi));
    if (!CreateProcessW(py, cmd, NULL, NULL, FALSE, 0, NULL, appdir, &si, &pi)) { _snwprintf(msg, 2 * N, L"Could not start the bundled Python (error %lu):\n%ls", GetLastError(), py); fail(msg); return 1; }
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    return 0;
}
