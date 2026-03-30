import ollama

def ask_rag_llm(user_query, vectorstore):
    # 1. Search the document for relevant snippets
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # 2. Strict System Prompt
    system_prompt = (
        "You are a private assistant. Answer the user's question ONLY using the provided context. "
        "If the answer is not in the context, say 'I do not have that information in my files.' "
        "Do not use your own knowledge. Keep it short for voice output. No markdown."
    )

    # 3. Combined Prompt
    full_prompt = f"Context: {context}\n\nQuestion: {user_query}"

    response = ollama.chat(
        model="gemma3:latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    
    # Sanitize for voice (as we did before)
    reply = response['message']['content']
    return reply.replace("*", "").strip()