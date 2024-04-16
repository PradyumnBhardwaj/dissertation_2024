
import webbrowser

def search_youtube(query):  # Accepts a query as a parameter 
    formatted_query = '+'.join(query.split())
    url = f"https://www.youtube.com/results?search_query={formatted_query}"
    webbrowser.open(url)
