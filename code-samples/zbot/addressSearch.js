const maxErrors = 1;

function search(address1, address2) {
    // Make capitalization consistent
    address1 = address1.toUpperCase();
    address2 = address2.toUpperCase();

    // Remove trailing spaces
    address1 = removeTrailing(address1);
    address2 = removeTrailing(address2);

    // Make abbreviations consistent
    address1 = abbreviate(address1);
    address2 = abbreviate(address2);

    // Run comparing function and return
    return compare(address1, address2);
}

function removeTrailing(address) {
    let addressSplit = address.split('');
    while (addressSplit[0] == " ") {
        addressSplit.splice(0, 1); // Delete the space
    }
    while (addressSplit[addressSplit.length - 1] == " ") {
        addressSplit.splice(addressSplit.length - 1, 1);
    }

    let finalAddress = "";
    for (let i = 0; i < addressSplit.length; i++) {
        finalAddress += addressSplit[i];
    }

    return finalAddress;
}

function abbreviate(address) {
    address = address.replace("LANE", "LN")
        .replace("STREET", "ST")
        .replace("AVENUE", "AVE")
        .replace("COURT", "CT")
        .replace("PLACE", "PL")
        .replace("DRIVE", "DR")
        .replace("CIRCLE", "CIR")
        .replace("CR", "CIR")
        .replace("TRAIL", "TRL")
        .replace("BOULEVARD", "BLVD")
        .replace("ROAD", "RD")
        .replace("NORTH", "N")
        .replace("SOUTH", "S")
        .replace("EAST", "E")
        .replace("WEST", "W")
        .replace("STATE", "ST")
        .replace("ROUTE", "RT")
        .replace("NORTHEAST", "NE")
        .replace("SOUTHEAST", "SE")
        .replace("NORTHWEST", "NW")
        .replace("SOUTHWEST", "SW")
        .replace("HIGHWAY", "HWY")
        .replace(",", "");

    return address;
}

function compare(address1, address2) {
    let address1Array = address1.split(" ");
    let address2Array = address2.split(" ");

    // Basically makes the larger array "leading" and smaller array "following"
    let leading = (address1Array.length >= address2Array.length) ? address1Array : address2Array; // Bigger array
    let following = (address1Array.length >= address2Array.length) ? address2Array : address1Array; // Smaller array
    for (let i = 0; i < following.length; i++) {
        for (let j = 0; j < leading.length; j++) {
            if (following[i] == leading[j]) {
                delete following[i]; // To signify this was found
                delete leading[j];  // So it can't be reused (that would be a weird scenario)
            }
        }
    }

    let errors = 0;
    for (let i = 0; i < following.length; i++) {
        if (following[i] != null) { // Wasn't found
            if (i == 0 || i == 1) { // First two items missed, usually HN and a street identifier
                return false; // Unacceptable to miss one of first two items
            }
            errors++;
        }
    }

    if (errors <= maxErrors) {
        return true;
    } else {
        return false;
    }
}

exports.search = search;