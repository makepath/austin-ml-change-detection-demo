import json
import os
import pathlib
import time
import numpy as np
import requests
from requests.auth import HTTPBasicAuth

PLANET_API_KEY = os.getenv('PLANET_KEY')
orders_url = 'https://api.planet.com/compute/ops/orders/v2'
auth = HTTPBasicAuth(PLANET_API_KEY,'')
headers = {'content-type':'application/json'}

same_src_products = [
	{
		"item_ids":["20190129_164426_1018", "20200201_163705_0e19"],
		"item_type": "PSScene",
		"product_bundle": "visual"
	}
]

def place_order(request, auth):
    response = requests.post(orders_url, data=json.dumps(request), auth=auth, headers=headers)
    print(response)
    
    if not response.ok:
        raise Exception(response.content)

    order_id = response.json()['id']
    print(order_id)
    order_url = orders_url + '/' + order_id
    return order_url

def poll_for_success(order_url, auth, num_loops=50):
    count = 0
    while(count < num_loops):
        count += 1
        r = requests.get(order_url, auth=auth)
        response = r.json()
        state = response['state']
        print(state)
        success_states = ['success', 'partial']
        if state == 'failed':
            raise Exception(response)
        elif state in success_states:
            break
        
        time.sleep(10)

def download_results(results, overwrite=False):
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    print('{} items to download'.format(len(results_urls)))
    
    for url, name in zip(results_urls,results_names):
        path = pathlib.Path(os.path.join('data',name))
    
        if overwrite or not path.exists():
            print('downloading {} to {}'.format(name,path))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
        else:
            print('{} already exists, skipping {}'.format(path,name))


request = {
"name":"2017_2022_AustinCD_test",
"products":same_src_products,
"metadata":{
	"stac":{
	}
},
"delivery":{"single_archive":True,
            "archive_type": "zip"
		}
}

order_url = place_order(request,auth)
poll_for_success(order_url,auth)
r = requests.get(order_url,auth=auth)
response = r.json()
results = response['_links']['results']
download_results(results)
