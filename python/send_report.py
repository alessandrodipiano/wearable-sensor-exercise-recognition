import csv
import io
import os
import smtplib
import threading
from email.message import EmailMessage

CLINICIAN_EMAIL = "stefanoanthony.rizzuto01@universitadipavia.it"


def _build_csv(profile: dict, pred_labels: list[str]) -> str:
    counts = {"correct": 0, "fast": 0, "low_amplitude": 0}
    for label in pred_labels:
        if label in counts:
            counts[label] += 1

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Name", "Surname", "Exercise", "Total Reps", "Correct", "Fast", "Low Amplitude"])
    writer.writerow([
        profile["name"],
        profile["surname"],
        profile["exercise"],
        len(pred_labels),
        counts["correct"],
        counts["fast"],
        counts["low_amplitude"],
    ])
    return buf.getvalue()


def _send(profile: dict, pred_labels: list[str]) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    sender = os.environ.get("GMAIL_USER", "")
    password = os.environ.get("GMAIL_APP_PASSWORD", "")
    if not sender or not password:
        return

    csv_content = _build_csv(profile, pred_labels)
    filename = f"report_{profile['name']}_{profile['surname']}.csv"

    msg = EmailMessage()
    msg["Subject"] = f"PT Session Report — {profile['name']} {profile['surname']}"
    msg["From"] = sender
    msg["To"] = CLINICIAN_EMAIL
    msg.set_content(
        f"Dear Clinician,\n\n"
        f"Please find attached the session report for "
        f"{profile['name']} {profile['surname']}.\n\n"
        f"Exercise: {profile['exercise']}\n"
        f"Total repetitions: {len(pred_labels)}\n"
        f"Correct: {sum(1 for l in pred_labels if l == 'correct')}\n"
        f"Too fast: {sum(1 for l in pred_labels if l == 'fast')}\n"
        f"Low amplitude: {sum(1 for l in pred_labels if l == 'low_amplitude')}\n\n"
        f"Best regards,\nPT Exercise Tracker"
    )
    msg.add_attachment(
        csv_content.encode("utf-8"),
        maintype="text",
        subtype="csv",
        filename=filename,
    )

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(msg)
    except Exception:
        pass


def send_report_background(profile: dict, pred_labels: list[str]) -> None:
    threading.Thread(target=_send, args=(profile, pred_labels), daemon=True).start()
