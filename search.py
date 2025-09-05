import os
from ollama import Client
from duckduckgo_search import DDGS

# pip install ollama duckduckgo-search
# ollama pull llama3.1:8b
# 위 두 command 실행 필요

HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
_OLLAMA_CLIENT = Client(host=HOST)

def search_internet(query):
    try:
        # 컨텍스트 매니저로 사용해 소켓/세션을 확실히 정리
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        # Filter out irrelevant results
        filtered_results = [result for result in results if 'body' in result]
        return filtered_results
    except Exception as e:
        print(f"Error searching internet: {e}")
        return []

def get_combined_response(query):
    # Preprocess user input
    query = query.strip().lower()

    # Perform internet search
    search_results = search_internet(query)
    search_content = "\n".join([result['body'] for result in search_results])[:2000]

    # Integrate search results into the AI response
    try:
        response = _OLLAMA_CLIENT.chat(
            model='llama3.1:8b',
            messages=[
                { 'role': 'system', 'content': (
                    '너는 한국어로 간결하게 답한다. '
                    '아래 검색 스니펫을 참고하되 사실만 정리하고, 가능하면 출처 URL을 함께 제시하라.'
                )},
                { 'role': 'user', 'content': f"질문: {query}\n\n검색 스니펫:\n{search_content}" },
            ]
        )
        return response['message']['content']
    except Exception as e:
        print(
            "Error generating AI response: {}\n".format(e)
            + "힌트: Ollama 서버가 실행 중인지 확인하세요 (기본 포트 11434).\n"
              f" - 현재 호스트: {HOST}\n"
              " - 서버 시작: ollama serve\n"
              " - 모델 다운로드: ollama pull llama3.1:8b\n"
        )
        return "Sorry, there was an error generating a response for the internet search."


query = "kt 주력 사업"
response = get_combined_response(query)

# Print the search results and the AI response
search_results = search_internet(query)
print("Search Results:")
for result in search_results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['href']}")
    print(f"Snippet: {result['body']}\n")

print("--------------------------------------------------------")
print("AI Response:")
print(response)

os.system('ollama stop llama3.1:8b')
