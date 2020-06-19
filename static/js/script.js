// Override defaiult behavior for input prediction submission using jquery and AJAX
$("#input-form").submit(function (e) {
    e.preventDefault(); // avoid to execute the actual submit of the form.

    var form = $(this);
    var url = form.attr('action');

    $.ajax({
        type: "POST",
        url: url,
        data: form.serialize(), // serializes the form's elements.
        success: function (data) {
            var pred_image = $("#pred_image");
            var slide_div = $("#slide-div");
            var pred_text = $("#pred_text");
            console.log('data: ' + data);

            // Path to image to display as output
            var img_path = ""
            var text = ""
            if (data == 'Negative') {
                img_path = "static/img/encolere.jpg"
                text = "Négatif!"
            } else if (data == 'Positive') {
                img_path = "static/img/heureux.jpg"
                text = "Positif!"
            } else if (data == 'Neutral') {
                img_path = "static/img/neutre.jpg"
                text = "Neutre!"
            } else {
                img_path = "static/img/pas_content.jpg"
                text= "Je ne comprends pas"
            }

            // First run, edge case of sliding div
            if (slide_div.is(":hidden")) {
                pred_image.attr("src", img_path);
            };

            // Toggle div showing for every subsequent run
            slide_div.slideToggle('slow', function () {
                if ($(this).is(":hidden")) {
                    $(this).slideToggle('slow')
                    pred_image.attr("src", img_path);
                };
            });

            pred_text.text(text)
        }
    });
});
