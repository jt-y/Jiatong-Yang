import os
import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Add your Slack Bot token here (from Slack API dashboard)
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")

client = WebClient(token=SLACK_BOT_TOKEN)

# List of chore people in rotation (Slack usernames or IDs)
# display_name = [xinyi, elvie, jiatong]
PEOPLE = ["U08SZSGDFTR", "U08TPA77ZFT", "U03EQ8Y1QSW"]

# Choose a fixed start date when rotation began
START_DATE = datetime.date(2025, 1, 6)  # Monday of first chore rotation

INTERVAL_DAYS = 14  # every two weeks

def get_chore_person():
    today = datetime.date.today()
    weeks_since = (today - START_DATE).days // INTERVAL_DAYS
    return PEOPLE[weeks_since % len(PEOPLE)]

def send_reminder():
    person_id = get_chore_person()
    message = f"🧹 Reminder: <@{person_id}>, you're on chores this week!"
    try:
        client.chat_postMessage(channel="#chores", text=message)
        print(f"Sent reminder: {message}")
    except SlackApiError as e:
        print(f"Error posting message: {e.response['error']}")

if __name__ == "__main__":
    send_reminder()
