import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--host', type = str, default= '')
parser.add_argument('--port', type = int, default= 0)
parser.add_argument('--user', type = str, default='')
parser.add_argument('--password', type = str, default='')


args = parser.parse_args()

host = args.host
port = args.port
user = args.user
password = args.password

