module.exports = class historyController{
    constructor() {

    }
    getHistoryTemplate() {
        return new Promise((resolve, reject) => {
            var historyCards = Handlebars.templates.historyCard()
            return resolve(historyCards)
        })
    }
}