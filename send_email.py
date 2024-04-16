
import smtplib
from email.message import EmailMessage
import ssl


# Prompting the user for email details
def send_email(receiver_email, subject, body):
    sender_email = "pradyumnbhardwaj@gmail.com"
    #receiver_email = input("Enter receiver's email address: ")
    password = 'YOUR PASSCODE HERE'
    # subject = input("Enter the subject of the email: ")
    # body = input("Enter the body of the email: ")

    email = EmailMessage()
    email['To'] = receiver_email
    email['From'] = sender_email
    email['Subject'] = subject
    email.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(sender_email, password)
        smtp.sendmail(sender_email, receiver_email, email.as_string())

    print("Email sent successfully!")

# text_mail = input('Write email: ')  

 
# import ollama         
# response_mail = ollama.chat(model='falcon:7b', messages=[
#     {'role': 'user', 'content': 'write an email on the topic' + text_mail},
# ])
#             # time.sleep(5)
# response_text_mail = response_mail['message']['content'] 
            


# def extract_subject(email_text: str) -> str:
#     """
#     Extracts the subject line from an email text.

#     Parameters:
#     - email_text (str): The text of the email.

#     Returns:
#     - str: The subject of the email.
#     """
#     lines = email_text.split("\n")
#     for line in lines:
#         if line.startswith("Subject:"):
#             return line.split("Subject: ")[1].strip()
#     return ""

# def extract_message(email_text: str) -> str:
#     """
#     Extracts the message body from an email text, excluding the subject line.

#     Parameters:
#     - email_text (str): The text of the email.

#     Returns:
#     - str: The message body of the email.
#     """
#     lines = email_text.split("\n")
#     message_body = ""
#     capture_message = False
#     for line in lines:
#         if capture_message:
#             message_body += line + "\n"
#         elif line.startswith("Subject:"):
#             capture_message = True
#     return message_body.strip()

# subject= extract_subject(response_text_mail)   
# print(subject)
# body= extract_message(response_text_mail)   
# print(body)    
# receiver='vishubhardwaj229@gmail.com'
# send_email(receiver, subject, body)

