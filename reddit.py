from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Chrome()

driver.get('https://www.reddit.com/r/wallstreetbets/?feedViewType=compactView')
assert 'Selenium' in driver.title

elem = driver.find_element(By.ID, 'absolute inset-0')
elem.click()
assert 'WebDriver' in driver.title

driver.quit()