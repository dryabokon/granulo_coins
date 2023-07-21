import os
import sys
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':


    K = [k for k in os.environ.keys()]
    V = [k for k in os.environ.values()]
    print(' '.join(K))

    for k,v in zip(K,V):
        print(k,v)

    # password = os.environ.get("USERNAME")
    # password = [p for p in password]
    # print(password)

    # password = os.environ.get("SECRET_LOGIN")
    #
    #
    # if(password=='Password123!'):
    #     print('A!!')
    # else:
    #     print('((')
