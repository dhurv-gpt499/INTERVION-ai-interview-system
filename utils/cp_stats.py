import requests
import json 

def fetch_leetcode_stats(username: str) -> dict:
    url = "https://leetcode.com/graphql"
    query = """
    {
      matchedUser(username: "%s") {
        submitStats {
          acSubmissionNum {
            difficulty
            count
          }
        }
        profile {
          ranking
          reputation
        }
      }
    }
    """ % username
    
    response = requests.post(url, json={"query": query})
    data = response.json()
    user = data["data"]["matchedUser"]
    
    stats = user["submitStats"]["acSubmissionNum"]
    return {
        "ranking": user["profile"]["ranking"],
        "easy_solved": stats[1]["count"],
        "medium_solved": stats[2]["count"],
        "hard_solved": stats[3]["count"],
        "total_solved": stats[0]["count"]
    }
    
def fetch_codeforces_stats(handle: str) -> dict:
    url = f"https://codeforces.com/api/user.info?handles={handle}"
    response = requests.get(url)
    data = response.json()
    
    if data["status"] == "OK":
        user = data["result"][0]
        return {
            "rating": user.get("rating", 0),
            "max_rating": user.get("maxRating", 0),
            "rank": user.get("rank", ""),
            "max_rank": user.get("maxRank", "")
        }
    return {}


if __name__ == "__main__":
    # Test with your actual profiles
    leet = fetch_leetcode_stats("dhruvgpt499-_-")
    print("=== LEETCODE ===")
    print(json.dumps(leet, indent=2))
    
    cf = fetch_codeforces_stats("Dhruv_Gupta422")
    print("\n=== CODEFORCES ===")
    print(json.dumps(cf, indent=2))