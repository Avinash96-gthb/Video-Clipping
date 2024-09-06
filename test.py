import ollama

res = ollama.chat(
	model="llava",
	messages=[
		{
			'role': 'user',
			'content': 'Describe this image',
			'images': ['images/frame_0329.png']
		}
	]
)

print(res['message']['content'])