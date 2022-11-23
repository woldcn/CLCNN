from datetime import datetime


class Log:
    def __init__(self, args):
        self.file = args.log_path
        self.str = ''
        self.print('.' * 50 + 'Start at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
        if args != None:
            for arg in dir(args):
                if arg[0] != '_':
                    self.print('{}: {}'.format(arg, getattr(args, arg)))

    def print(self, str):
        print(str)
        self.str += str + '\n'

    def save(self):
        self.print('.' * 50 + 'End at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
        with open(self.file, 'w') as f:
            print(self.str, file=f)
