import os
import numpy as np
import requests
import argparse
import json
import time
import logging
import csv

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL