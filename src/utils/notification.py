
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging
from typing import Optional, List
import os

logger = logging.getLogger(__name__)

class EmailNotifier:
    def __init__(self, config: dict):
        """
        Initialize the EmailNotifier.
        
        Args:
            config: Dictionary containing email configuration:
                - smtp_server
                - smtp_port
                - sender_email
                - sender_password
                - receiver_emails (list or comma-separated string)
                - cooldown_seconds (default: 300)
        """
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.sender_email = config.get('sender_email', '')
        self.sender_password = config.get('sender_password', '')
        
        receivers = config.get('receiver_emails', [])
        if isinstance(receivers, str):
            self.receiver_emails = [r.strip() for r in receivers.split(',')]
        else:
            self.receiver_emails = receivers
            
        self.cooldown_seconds = config.get('cooldown_seconds', 300) # 5 minutes default
        self.last_alert_time = 0
        
    def send_fire_alert(self, confidence: float, image_path: Optional[str] = None, location: str = "Unknown"):
        """
        Send a fire alert email.
        
        Args:
            confidence: Confidence score of the detection (0-1).
            image_path: Path to the image file showing the detection (optional).
            location: Description of the location (optional).
        """
        if not self.sender_email or not self.receiver_emails:
            logger.warning("Email configuration missing. Alert not sent.")
            return

        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown_seconds:
            logger.info(f"Alert suppressed due to cooldown. Next alert allowed in {int(self.cooldown_seconds - (current_time - self.last_alert_time))}s.")
            return

        subject = f"🔥 FIRE DETECTED! (Confidence: {confidence:.2%})"
        body = f"""
        URGENT: Fire detected by the monitoring system.
        
        Confidence: {confidence:.2%}
        Location: {location}
        Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
        
        Please verify immediately.
        """

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ", ".join(self.receiver_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)
            except Exception as e:
                logger.error(f"Failed to attach image: {e}")

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info("Fire alert email sent successfully.")
            self.last_alert_time = current_time
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
