# This script requires 'selenium' and 'webdriver-manager'
# Install them with: pip install selenium webdriver-manager

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def run_login_test():
    """
    This is a TRADITIONAL selenium script.
    It is brittle: if the ID 'username' or 'password' changes,
    this test will break.
    """
    print("--- Task 2: Automated Testing ---")
    
    # Use webdriver-manager to automatically handle the driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    # Target site for demo
    url = "http://quotes.toscrape.com/login"
    driver.get(url)
    
    try:
        # --- Test 1: Invalid Login ---
        print("Running test 1: Invalid Credentials")
        driver.find_element(By.ID, "username").send_keys("wronguser")
        driver.find_element(By.ID, "password").send_keys("wrongpass")
        driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()
        
        # Check for error message
        error_element = driver.find_element(By.CSS_SELECTOR, ".alert-danger")
        if "Invalid username or password" in error_element.text:
            print("  [SUCCESS] Invalid login test passed.")
        else:
            print("  [FAILURE] Invalid login test failed.")

        time.sleep(1)

        # --- Test 2: Valid Login ---
        print("Running test 2: Valid Credentials")
        # Clear fields
        driver.find_element(By.ID, "username").clear()
        driver.find_element(By.ID, "password").clear()
        
        # NOTE: This site has no real login. We just test the UI.
        # For a real test, you'd use 'user' and 'pass'.
        driver.find_element(By.ID, "username").send_keys("user")
        driver.find_element(By.ID, "password").send_keys("pass")
        driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()
        
        # Check for success (e.g., redirect to main page and see "Logout" link)
        logout_link = driver.find_element(By.LINK_TEXT, "Logout")
        if logout_link.is_displayed():
            print("  [SUCCESS] Valid login test passed.")
        else:
            print("  [FAILURE] Valid login test failed.")
            
    except Exception as e:
        print(f"  [ERROR] Test execution failed: {e}")
    
    finally:
        print("Test run complete. Closing browser.")
        driver.quit()

if __name__ == "__main__":
    run_login_test()
