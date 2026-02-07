import sys, os

if __name__ == "__main__":
    print("Hello World!")
    print("cwd:", os.getcwd())
    print("first paths: ", sys.path)
    print(os.environ.get("PYTHONPATH"))
