from agents import app

result = app.invoke({
    "messages": {
        "role": "user",
        "content": "What is the sum of 5 and 10?"
    }
})

for r in result["messages"]:
    r.pretty_print()
