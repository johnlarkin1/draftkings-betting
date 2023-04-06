function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function cleanseUnicodeMinus(exampleStr) {
  return exampleStr.replace(String.fromCharCode(8722), '-');
}

async function scrollToBottom() {
  for (let i = 0; i < 10; i++) {
    var bottom = document.querySelector('footer[class="sportsbook-footer"]');
    bottom.scrollIntoView({ behavior: 'smooth' });
    await sleep(2000);
  }
}

async function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
  await sleep(2000);
}

function cleanseCommas(str) {
  return str.replaceAll(', ', '+');
}

function cleanseStake(stake) {
  return '$' + stake.split('$')[1];
}

function parseBetLogic(actualBet) {
  console.log('actualBet.nodeType', actualBet.nodeType);
  console.log('actualBet', actualBet);
  var title = cleanseCommas(actualBet.querySelector('span[data-test-id^="bet-details-title"]').textContent);
  var attemptDisplayOdds = actualBet.querySelector('span[data-test-id^="bet-details-displayOdds"]');
  if (!attemptDisplayOdds) {
    // We most likely have some adjusted bets
    var originalOdds = actualBet.querySelector('span[data-test-id^="bet-details-original-displayOdds"]');
    var boostedOdds = actualBet.querySelector('span[data-test-id^="bet-details-boosted-displayOdds"]');
  }
  var displayOdds =
    attemptDisplayOdds?.textContent ?? boostedOdds?.textContent ?? originalOdds?.textContent ?? 'unknown';
  var subtitle = cleanseCommas(actualBet.querySelector('span[data-test-id^="bet-details-subtitle"]').textContent);

  // if parlay is in the title, then we want to flip subtitle and title
  if (title.toLowerCase().includes('parlay')) {
    var temp = title;
    title = subtitle;
    subtitle = temp;
  }
  var status = actualBet.querySelector('div[data-test-id^="bet-details-status"]').innerText;
  var cleansedStake = cleanseStake(actualBet.querySelector('span[data-test-id^="bet-stake"]').textContent);
  var returnsNode = actualBet.querySelector('span[data-test-id^="bet-returns"]');
  var returns = returnsNode ? cleanseStake(returnsNode.textContent) : `-$${parseFloat(cleansedStake.slice(1))}`;

  // Parlays have bet-details instead of event-reference
  var potentialBetDetails = actualBet.querySelectorAll('span[data-test-id^="bet-reference"]');
  var betTime = 'unknown';
  if (potentialBetDetails.length > 0) {
    var betTime = potentialBetDetails[0].textContent;
  }

  return { title, displayOdds, subtitle, status, cleansedStake, returns, betTime };
}

async function getBetData() {
  // {
  //   title: '',
  //   displayOdds: '',
  //   subtitle: '',
  //   status: '',
  //   stake: '',
  //   returns: '',
  //   betTime: '',
  // },
  var betDetails = [];
  let keepGoing = true;
  var bets = document.querySelector('sb-lazy-render[data-testid="sb-lazy-render"]');
  if (bets) {
    do {
      var child = bets.firstElementChild;
      var actualBets = child.childNodes;
      for (const actualBet of actualBets) {
        // It has to be an element_node
        if (actualBet.nodeType == Node.ELEMENT_NODE) {
          var betDetail = parseBetLogic(actualBet);
          betDetails.push(betDetail);
        }
      }

      // Scroll to bottom of that div to see if we need to load more
      var bets = bets.nextElementSibling;
      bets.scrollIntoView({ behavior: 'smooth' });
      await sleep(2000);
      keepGoing = bets.hasChildNodes();
      // if we don't have any children, try one more time to reload
      if (!keepGoing) {
        bets = document.querySelector('sb-lazy-render[data-testid="sb-lazy-render"]');
        keepGoing = bets.hasChildNodes();
      }
    } while (keepGoing);
  }
  return betDetails;
}

function convertToCSVData(betData) {
  let csvContent = Object.keys(betData[0]).join(',') + '\n';
  betData.forEach((item) => {
    var strRow = Object.values(item).join(',');
    strRow = cleanseUnicodeMinus(strRow);
    csvContent += strRow + '\n';
  });
  return csvContent;
}

async function getFileHandle() {
  const currDate = new Date().toLocaleDateString('en-US').replace('-', '_').replaceAll('/', '_');
  const options = {
    types: [
      {
        description: 'Text Files',
        suggestedName: `DraftKings${currDate}`,
        accept: {
          'text/plain': ['.csv'],
        },
      },
    ],
  };
  const handle = await window.showSaveFilePicker(options);
  return handle;
}

async function writeCSVFile(csvData, handle) {
  const writable = await handle.createWritable();
  await writable.write(csvData);
  await writable.close();
  return;
}

chrome.runtime.onMessage.addListener(async function (request, sender, sendResponse) {
  if (request.method == 'scrapePage') {
    var handle = await getFileHandle();
    await scrollToBottom();
    await scrollToTop();
    var betData = await getBetData();
    var csvData = convertToCSVData(betData);
    await writeCSVFile(csvData, handle);
    sendResponse({ text: document.body.innerText, method: 'scrapePage' });
    alert(`Congrats! Draftkings data scraped to ${handle.name}`);
  }
});
