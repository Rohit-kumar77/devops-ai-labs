def read_log_file(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            return lines
    except FileNotFoundError:
        print(f"Log file not found: {file_path}")
        return []


def main():
    log_file = "../logs/sample.log"
    logs = read_log_file(log_file)

    print("---- LOG FILE CONTENT ----")
    for line in logs:
        print(line.strip())


if __name__ == "__main__":
    main()
