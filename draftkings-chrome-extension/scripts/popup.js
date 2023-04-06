function setUp() {
  var checkButton = document.getElementById('scrape-button');
  checkButton.addEventListener(
    'click',
    function () {
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        chrome.tabs.sendMessage(tabs[0].id, { method: 'scrapePage' }, function (response) {
          if (response.method == 'scrapePage') {
            alert('Successfully scraped Draftkings! Wrote file locally to ' + response.text);
          }
        });
      });
    },
    false
  );
}

document.addEventListener('DOMContentLoaded', setUp, false);
