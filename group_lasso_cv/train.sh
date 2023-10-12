#!/usr/bin/env sh
exec 1>log_grouplasso_find_e.txt
/usr/bin/env python grouplasso_find_e.py
exec 1>log_grouplasso_find_lambda.txt
/usr/bin/env python grouplasso_find_lambda.py
exit 0