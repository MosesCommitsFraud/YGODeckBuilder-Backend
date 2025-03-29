import openai

# Konfiguriere die Azure OpenAI Details
openai.api_type = "azure"
openai.api_base = "https://miker-m83k3q0u-swedencentral.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
openai.api_key = "6mj1VsQqmAoXcfbyO3siypqOaiYWe2aLIhWRFZ8WRXv6GkBPlwlFJQQJ99BCACfhMk5XJ3w3AAAAACOGiZYy"

# Setze den Namen des Deployments (entspricht dem Modellnamen)
deployment_name = "gpt-4"

# Beispiel: Sende eine Chat-Anfrage an das Modell
response = openai.ChatCompletion.create(
    engine=deployment_name,
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Hallo, wie kannst du mir helfen?"},
    ],
    max_tokens=150
)

# Ausgabe der Antwort
print(response["choices"][0]["message"]["content"])
