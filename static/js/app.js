function setMode(mode){

    fetch("/set_mode/" + mode)
    .then(response => response.json())
    .then(data => {

        document.getElementById("mode").innerHTML =
            "Mode : " + data.mode;

        console.log("Mode aktif:", data.mode);

    })
    .catch(error => {

        console.error(error);

    });

}

function updatePrediction(){

    fetch("/prediction")
    .then(response => response.json())
    .then(data => {

        document.getElementById("prediction").innerHTML =
            data.prediction;

        document.getElementById("mode").innerHTML =
            "Mode : " + data.mode;

        let conf = data.confidence || 0;

        let bar =
            document.getElementById("confidenceBar");

        bar.style.width = conf + "%";
        bar.innerHTML = conf + "%";

        if(conf >= 80){

            bar.className =
            "progress-bar bg-success progress-bar-striped progress-bar-animated";

        }
        else if(conf >= 50){

            bar.className =
            "progress-bar bg-warning progress-bar-striped progress-bar-animated";

        }
        else{

            bar.className =
            "progress-bar bg-danger progress-bar-striped progress-bar-animated";

        }

    })
    .catch(error => {

        console.error(error);

    });

}

updatePrediction();

setInterval(updatePrediction, 300);