import re


DANGEROUS_FUNCTIONS = [
    "base64_encode", "base64_decode", "eval", "str_replace", "assert",
    "system", "cmd_shell", "fopen", "fwrite", "pcntl", "c99_buff_prepare",
    "shell_exec", "shell", "exec", "curl_exec", "proc_open", "python_eval",
    "file_get_contents", "file_put_contents", "curl", "popen", "include",
    "require", "include_once", "array_map", "array_walk", "posix_getpwuid",
    "fileowner", "filegroup", "posix_getgrgid", "str_rot13", "gzencode",
    "gzdeflat", "gzcompress", "passthru", "unserialize", "xpath_eval",
    "get_headers", "get_browser", "fgets", "dlob", "readdir",
    "mysql_fetch_array", "mysql_fetch_object", "c99sh_surl",
]

DANGEROUS_COMMANDS = [
    "wget", "lynx", "get", "fetch", "perl", "python", "gcc", "chmod",
    "nohup", "nc", "uname", "id", "ver", "sysctl", "whoami", "pwd",
]

DANGEROUS_VARIABLES = [
    "$ostype", "$_get", "$_post", "$_cookie", "$_request",
    "$_files", "$_session", "$cmd", "$_server", "$_env",
]

DANGEROUS_KEYWORDS = [
    "webshell by", "web shell by", "hack by", "bypass AV", "developed by",
    "password is", "r57", "c99", "n3shell", "tryang team", "c99shell",
    "cod3rz", "xakep", "http://cctea-m.ru/update/c999shell",
    "http://ccteam.ru/files/c999sh_sources", "phpspy",
]


def _count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(re.escape(pattern), text, re.IGNORECASE))


class LexicalFeatures:

    def extract(self, text: str) -> dict:
        features = {}
        lower = text.lower()
        for fn in DANGEROUS_FUNCTIONS:
            features[fn] = _count_pattern(lower, fn)
        for cmd in DANGEROUS_COMMANDS:
            features[cmd] = _count_pattern(lower, cmd)
        for var in DANGEROUS_VARIABLES:
            features[var] = _count_pattern(lower, var)
        for kw in DANGEROUS_KEYWORDS:
            features[kw] = _count_pattern(lower, kw)
        return features

    @staticmethod
    def all_feature_names() -> list:
        return (
            DANGEROUS_FUNCTIONS
            + DANGEROUS_COMMANDS
            + DANGEROUS_VARIABLES
            + DANGEROUS_KEYWORDS
        )
