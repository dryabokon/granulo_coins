import os
import sys
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    print(sys.argv)
    # print(' '.join([k for k in os.environ.keys()]))
    print(os.environ.get("SECRET_LOGIN"))
