const uiController = require("./uiController");

module.exports = class processController{
    constructor() {
        window.uiManager = new uiController()
    }
    startProcess() {
        console.log("process started")
        console.log("scanner started")
        window.uiManager.prepare()
        .then( result => {
             console.log("UI prepared")
        })
    }
}