
from offline_ai_assistant.assistant.chat import ChatEngine


def main():
    chat = ChatEngine()

    print("\nOffline AI Assistant")
    print("Type /reset to clear memory | /exit to quit")
    print("-" * 45)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "/exit":
            print("Goodbye!")
            break

        if user_input == "/reset":
            chat.reset()
            print("Conversation history cleared!")
            continue

        if not user_input:
            continue

        try:
            response = chat.send_message(user_input)
            print(f"\nAssistant: {response}")

        except RuntimeError as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
