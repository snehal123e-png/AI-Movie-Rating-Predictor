const form =
    document.getElementById(
        "predictionForm"
    );

const result =
    document.getElementById(
        "result"
    );

const rating =
    document.getElementById(
        "rating"
    );

const category =
    document.getElementById(
        "category"
    );

const message =
    document.getElementById(
        "message"
    );

const predictButton =
    document.getElementById(
        "predictButton"
    );


const API_URL =
    "https://YOUR-RENDER-BACKEND.onrender.com/predict";


form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();

        predictButton.disabled = true;

        predictButton.textContent =
            "🤖 Predicting...";


        result.classList.remove(
            "hidden"
        );

        rating.textContent =
            "...";


        category.textContent =
            "Processing";


        message.textContent =
            "Sending movie details to the AI model...";


        const movieData = {

            genre:
                document
                    .getElementById("genre")
                    .value,

            runtime:
                Number(
                    document
                        .getElementById("runtime")
                        .value
                ),

            budget:
                Number(
                    document
                        .getElementById("budget")
                        .value
                ),

            year:
                Number(
                    document
                        .getElementById("year")
                        .value
                ),

            votes:
                Number(
                    document
                        .getElementById("votes")
                        .value
                ),

            gross:
                Number(
                    document
                        .getElementById("gross")
                        .value
                )
        };


        try {

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                movieData
                            )
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Prediction failed"
                );
            }


            const data =
                await response.json();


            rating.textContent =
                `${Number(
                    data.predicted_rating
                ).toFixed(2)} / 10`;


            category.textContent =
                data.category;


            message.textContent =
                data.message;


        } catch (error) {

            rating.textContent =
                "Error";


            category.textContent =
                "Unable to predict";


            message.textContent =
                "Please check whether the Render backend is running and the API URL is correct.";

            console.error(error);

        } finally {

            predictButton.disabled =
                false;

            predictButton.textContent =
                "🤖 Predict IMDb Rating";
        }
    }
);