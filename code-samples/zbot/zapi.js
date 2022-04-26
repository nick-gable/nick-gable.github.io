const got = require('got');
const addressSearch = require('./addressSearch');

const requestUrl = "https://api.bridgedataoutput.com/api/v2/zestimates?access_token={T}&limit={L}&near={A}";
const accessToken = "a28cc8ddb645628094d2c25ec56a7f13";

function getZestimate(address, limit, callback) { // For this to work, format address as HN ST CITY STATE ZIP
    (async () => {
        let request = requestUrl.replace("{A}", address)
            .replace("{T}", accessToken)
            .replace("{L}", limit);

        const response = await got(request);
        let responseObject = JSON.parse(response.body);
        if (responseObject.success) {
            for (let i = 0; i < responseObject.bundle.length; i++) {
                if (responseObject.bundle[i].address.toUpperCase().includes(address.toUpperCase())) {
                    let successObject = responseObject.bundle[i];
                    callback((successObject.zestimate == null) ? "X" : successObject.zestimate
                        , (successObject.zillowUrl == null) ? "X" : successObject.zillowUrl);
                    return;
                }
            }

            // Strict search failed, use search algorithm and attempt again
            for (let i = 0; i < responseObject.bundle.length; i++) {
                if (addressSearch.search(address, responseObject.bundle[i].address)) { // Algorithm determined as same
                    let successObject = responseObject.bundle[i];
                    callback((successObject.zestimate == null) ? "X" : successObject.zestimate
                        , (successObject.zillowUrl == null) ? "X" : successObject.zillowUrl);
                    return;
                }
            }

            // If we get here, there were no successful matches
            if (limit >= 35) { // This means there were no results in 35 found - probably a typo
                // HN search has proven inaccurate
                console.log("Failure");
                let houseNumber = address.split(" ")[0];
                for (let i = 0; i < responseObject.bundle.length; i++) {
                    if (responseObject.bundle[i].address.includes(houseNumber)) {
                        let successObject = responseObject.bundle[i];
                        console.log("HN found address: " + successObject.address);
                        //callback((successObject.zestimate == null) ? "X" : successObject.zestimate
                        //    , (successObject.zillowUrl == null) ? "X" : successObject.zillowUrl); // ! since we used HN search
                        //return;
                    }
                }
                callback("?", "?");
            } else { // Weird, but not in first five
                getZestimate(address, limit + 5, callback);
            }
        } else {
            callback("ER", "ER");
            console.log(response);
        }
    })();
}

function testBack(zestimate, url) {
    console.log("Zestimate: " + zestimate);
    console.log("URL: " + url);
}

exports.getZestimate = getZestimate;
exports.testBack = testBack;