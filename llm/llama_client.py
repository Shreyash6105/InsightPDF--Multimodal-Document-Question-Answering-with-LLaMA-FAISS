import subprocess

# def call_llama(prompt: str) -> str:
#     #Sends a prompt to LLaMA via Ollama and returns the response text.
#     result = subprocess.run(
#         ["ollama", "run", "llama3"],
#         input=prompt,
#         text=True,
#         capture_output=True  #capture o/p (stdout) instead of directly showing it to terminal
#     )
#     return result.stdout.strip()  #strip removes leading or trailing whitespaces or new lines



def call_llama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",     # FORCE UTF-8
        errors="ignore"       # ignore bad bytes
    )
    return result.stdout.strip()
