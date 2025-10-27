import requests


def fetch_observations(taxon_id: int, page_number: int, images_per_page) -> list:
    """
    Fetches observations for a given taxon ID and page from the iNaturalist API.

    Args:
        taxon_id (int): The taxon ID to query.
        page_number (int): The page number for pagination.

    Returns:
        list: A list of observation dictionaries.
    """
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "has[]": "photos",
        "geo": "true",
        "page": page_number,
        "per_page": images_per_page,
        "order_by": "votes",
        "quality_grade": "research",
        "photo_license": "cc0,cc-by",
    }
    response = requests.get(url, params=params)
    print(f"Requesting {url} with params {params}")
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        print(f"Erreur API (taxon {taxon_id}, page {page_number})")
        return []
