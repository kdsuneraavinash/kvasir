from termcolor import colored


class Printer:
    @staticmethod
    def information(message: str):
        print(colored(f"[INFO] {message}", 'cyan'))

    @staticmethod
    def warning(message: str):
        print(colored(f"[WARNING] {message}", 'yellow'))

    @staticmethod
    def error(message: str):
        print(colored(f"[ERROR] {message}", 'white', 'on_red'))
        exit()

    @staticmethod
    def processing(message: str):
        print(colored(f"[PROCESSING] {message}", 'green'), end='\r')

    @staticmethod
    def default(message: str):
        print(message)

    @staticmethod
    def end_processing():
        print()
