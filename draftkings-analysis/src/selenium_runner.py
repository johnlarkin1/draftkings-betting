"""A partial attempt to use Selenium to access the Draftkings MyBet section"""

import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
options.add_argument("start-maximized")
options.add_argument("disable-web-security")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
driver = webdriver.Chrome(
    options=options, service=Service(ChromeDriverManager().install())
)

TARGET_URL = "https://sportsbook.draftkings.com/mybets"
driver.get(TARGET_URL)
driver.find_element("xpath", '//button[@data-test-id="notification-button"]').click()

# Enter credentials
username = driver.find_element("xpath", '//input[@id="login-username-input"]')
username.send_keys("john@johnjlarkin.com")
password = driver.find_element("xpath", '//input[@id="login-password-input"]')
password.send_keys("<insert your password here>")

# Login
driver.find_element("xpath", '//button[@id="login-submit"]').click()
time.sleep(100)
