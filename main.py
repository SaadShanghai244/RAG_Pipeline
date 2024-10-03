from app.retrieval_object import TextEmbeddings
import asyncio



async def main():
    user_input = input("It must be doc or scrape \n")
    query = input("Type in your query: \n")
    while query != "exit":
        obj = TextEmbeddings(query)
        await obj.main(user_input=user_input,query=query)  # Ensure the path to the PDF file is correct
        query = input("Type in your query: \n")

if __name__ == "__main__":
    asyncio.run(main())