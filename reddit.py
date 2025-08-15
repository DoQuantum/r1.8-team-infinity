from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()

driver.get('https://old.reddit.com')
assert 'reddit' in driver.title

elem = driver.find_element(By.ID, 'thing_t3_1mco73c')
elem.click()
assert 'WebDriver' in driver.title

driver.quit()