import requests


class ApiBaseRepository:

    def __init__(self, base_url, headers=None):
        self.base_url = base_url
        self.headers = headers

    def get_url(self, endpoint="", query_str=""):
        if query_str:
            query_str = f"?{query_str}"
        return f"{self.base_url}{endpoint}{query_str}"

    def get(self, endpoint="", query_str=""):
        url = self.get_url(endpoint, query_str)
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint, data_dict):
        url = self.get_url(endpoint)
        response = requests.post(url, headers=self.headers, json=data_dict)
        response.raise_for_status()
        return response.json()

    def put(self, endpoint, data_dict):
        url = self.get_url(endpoint)
        response = requests.put(url, headers=self.headers, json=data_dict)
        response.raise_for_status()
        return response.json()

    def delete(self, endpoint="", query_str=""):
        url = self.get_url(endpoint, query_str)
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

