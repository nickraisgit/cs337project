from google import genai

def test_gemini():
    try:
        client = genai.Client(api_key="AIzaSyDZzNJw90nAlyRdQ4Me8GObBp9keMKWg68")
        models = client.models.list()
        for m in models:
            print(m.name)
        response = client.models.generate_content(
            model="gemini-pro-latest", 
            contents="Say 'Gemini API is working!'"
        )

        print("Response:", response.text)

    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    test_gemini()