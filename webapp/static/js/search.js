function searchForTerm(){
    let searchTerm = document.getElementById("searchTerm").value;
    window.location.href = "/suche/" + searchTerm;
}