var debug = true;
var is_chief = (navigator.userAgent.indexOf("Chrome") == -1) && (navigator.userAgent.indexOf("Firefox") == -1);
// var is_chief = window.location.search.substring(1).indexOf("chief") >= 0;
// the_gs is mainly for chief, it is where it keeps the game state.
// Non-chiefs use it only for two reasons:
//    - To detect the first message from fibbage/gamestate: so we always display
//		(otherwise, if we reload the page and the last gamestate is non-refreshing
//		we would see nothing). the_gs is null on reload, that's how we detect the
//		first message.
//	  - For certain callbacks that need the current gamestate, even if there was
//		no new display to update the callback itself. E.g., the proposing and voting
//		callbacks that need to know how many people have proposed or voted so far,
//		but won't see the update because they are non-refreshing.
var the_gs = null;
var questions = null
var final_questions = null

var diffusion_params = {
		host   : 'klander-l102aw.eu.diffusion.cloud',
		port   : 443,
		secure : true,
		principal : 'admin',
		credentials : 'admin'
	}
var TopicSpecification = diffusion.topics.TopicSpecification;
var TopicType = diffusion.topics.TopicType;


var add_player_topic_callback = function(path, newValue, player_names) {
	if (debug) console.log('Got string update for topic: ' + path, newValue);
	var player_name = newValue;
	if (player_names.indexOf(player_name) == -1) {
		player_names.push(player_name);
		if (debug) console.log("Added player " + player_name + ", players are " + player_names);
	} else {
		if (debug) console.log("Player " + player_name + "repeated, players are " + player_names);
	}
	$(".manage p").text("Jugadores: " + player_names);
}


var start_button_click_callback = function(session, player_names) {
	if (player_names.length <2) {
		alert("Cannot play with less than 2 players.");
		return;
	}
	the_gs = new GameState(questions.length, final_questions.length, player_names)
	chief_start_game(session);
	execute_command(session, the_gs, {"name": "init"});
	setCookie("game_is_on", true, 3600 * 6);  // game on for 6 hours
}


var send_player_name = function(session, player_name) {
	if (debug) console.log("Sending player name " + player_name);
	setCookie("player_name", player_name, 3600 * 6);
	session.topics.updateValue('fibbage/player_names',
							   player_name,
							   diffusion.datatypes.string());
	if (!is_chief) subscribe_to_gamestate(session, player_name);
	$("div.waiting_for_session").show();
	$("div.player_name").hide();
}


var setup_player_name = function(session) {
	player_name = getCookie("player_name");
	$("div.game_container").hide();
	$("div.manage_container").show();
	if (!player_name) {
		if (debug) console.log("No player name registered yet.");
		$("div.waiting_for_session").hide();
		$("div.player_name").show();
		$(".player_name_form").on("submit", function(event){
			event.preventDefault();
			if($(".player_name_form input").val()) {
				$(".player_name_form input").prop("disabled", true);
				send_player_name(session, $(".player_name_form input").val());
			}
		});
	} else {
		send_player_name(session, player_name);
	}
}


var setup_game = function(session) {
	var player_names = [];
	return session.topics.remove('fibbage/gamestate')
	.then(() => {
		console.log('Removed topic fibbage/gamestate');
		}, reason => {
		console.log('Failed to remove topic fibbage/gamestate: ', reason);
	})
	.then(() => {
		return session.topics.add('fibbage/player_names',
								  new TopicSpecification(TopicType.STRING, {DONT_RETAIN_VALUE: "true"}))})
	.then(result => {
		console.log('Added topic: ' + result.topic);
		session
			.addStream('fibbage/player_names', diffusion.datatypes.string())
			.on('value', function(path, specification, newValue, oldValue) {
				add_player_topic_callback(path, newValue, player_names);
			});
		// Subscribe to the topic. The value stream will now start to receive notifications.
		session.select('fibbage/player_names');
		}, reason => {
			console.log('Failed to add topic: ', reason);
		})
	.then(() => {
		$(".manage button").click(() => {start_button_click_callback(session, player_names);});
		return session;
	});
}

var subscribe_to_gamestate = function(session, player_name) {
    session
        .addStream('fibbage/gamestate', diffusion.datatypes.json())
        .on('value', function(path, specification, newValue, oldValue) {
            var is_first = (the_gs === null);
            var gs = JSON.parse(newValue.get());
            var refresh = gs.refresh;
            if (debug) console.log('Got JSON update for topic: ' + path, gs);
            if (is_first) {
                the_gs = new GameState(gs.n_questions, gs.n_final_questions, gs.player_names);
            }
            the_gs = Object.assign(the_gs, gs);
            if (refresh || is_first) display(session, gs, player_name);
            else {
                if (gs.stage == 1) display_missing_votes(gs, player_name);
            }
            if (is_first && is_chief) {
                if (debug) console.log("Subscribing to fibbage/command after receiving fibbage/gamestate, the_gs is ", the_gs);
                subscribe_to_command(session, the_gs);
            }
        });
    // Subscribe to the topic. The value stream will now start to receive notifications.
    session.select('fibbage/gamestate');
}

var subscribe_to_command = function(session, gs) {
	if (debug) console.log("Subscribing to fibbage/command, gs is ", gs);
	session
		.addStream('fibbage/command', diffusion.datatypes.json())
		.on('value', function(path, specification, newValue, oldValue) {
			if (debug) console.log('Got JSON update for topic: ' + path, newValue.get());
			execute_command(session, gs, JSON.parse(newValue.get()));
		});
	// Subscribe to the topic. The value stream will now start to receive notifications.
	session.select('fibbage/command');
}


var chief_start_game = function(session) {
	session.topics.add('fibbage/gamestate',
					   new TopicSpecification(TopicType.JSON)).then(function(result) {
        console.log('Added topic: ' + result.topic);
		subscribe_to_gamestate(session, getCookie("player_name"));
    }, function(reason) {
        console.log('Failed to add topic: ', reason);
	});

    session.topics.add('fibbage/command',
					   new TopicSpecification(TopicType.JSON,
											  {DONT_RETAIN_VALUE: "true"})).then(function(result) {
        console.log('Added topic: ' + result.topic);
		// When called from the start button, the chief has the_gs already
		// and can and should start to listen to commands.
		// (The alternative is starting from reload, we will get
		// the gs on the first message after subscribing to fibbage/game_state)
		if (the_gs) {
			if (debug) console.log("Subscribing to fibbage/command " +
								   "on start game, the_gs is " + the_gs);
			subscribe_to_command(session, the_gs);
		}
    }, function(reason) {
        console.log('Failed to add topic: ', reason);
	});
}

var pub_command = function(session, obj) {
	if (!session) {
		console.log("No session, couldn't pub command " + obj);		
	} else {
		if (debug) {
			console.log("Publishing command with name " + obj["name"])
		}
		session.topics.updateValue('fibbage/command',
								   JSON.stringify(obj),
								   diffusion.datatypes.json());
	}
}

var execute_command = function(session, gs, obj) {
	if (debug) {
		console.log("Executing command, name=" + obj["name"] + ", arg1=" + obj["arg_1"] + ", arg2=" + obj["arg_2"]);
		console.log("the_gs is ", the_gs);
	}
	var refresh = false;
	if (obj["name"] == "init") {
		refresh = gs.init();
	}
	if (obj["name"] == "propose") {
		refresh = gs.propose(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "vote") {
		refresh = gs.vote(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "next_round") {
		refresh = gs.next_round();
	}
	gs.refresh = refresh;
	session.topics.updateValue('fibbage/gamestate',
							   JSON.stringify(gs),
							   diffusion.datatypes.json());
}


var display_voted = function(gs, voted_whom) {
	var window_width = $(window).width();
	var card_height = $("div.candidates").height();
	var card_width = card_height * 11 / 16;
	card_width = Math.min(
		card_width,
		(window_width - card_width) / (gs.n_players - 1));
	var cum_votes = [];
	for (var i = 0; i < gs.candidates_shown.length; i++)
		cum_votes.push(0);
	var cum_votes_so_far = cum_votes.slice();
	for (var i=0; i<gs.n_players; i++) {
		if (voted_whom[i] == -1) continue;
		cum_votes[voted_whom[i]]++;
	}
	console.log("Cum votes:" + cum_votes + ", voted_whom:" + voted_whom);
	for (var i=0; i<gs.n_players; i++) {
		idx = voted_whom[i];
		if (idx == -1) continue;
		if (gs.candidates_shown[idx] == gs.candidates[gs.turn]) color="blue";
		else color="red";
		$(".candidates p.voted").eq(i).show()
			.text(gs.player_names[i])
			.css("color", color)
			.css("left", idx * card_width)
			.css("top", 0.3 * card_height + 0.7 * card_height * cum_votes_so_far[idx] / cum_votes[idx])
			.outerWidth(card_width, true);
		cum_votes_so_far[idx]++;
	}	
	for (var i=gs.n_players; i<$(".candidates p.voted").length; i++) {
		$(".candidates p.voted").eq(i).hide().text("");
	}
}

var display_missing_votes = function(gs, player_name) {
	var player_idx = gs.player_names.indexOf(player_name);
	var missing = [];
	for (var i = 0; i<gs.n_players; i++) {
		if (gs.votes[i] == -1) missing.push(gs.player_names[i]);
	}
	var missing_text_1 = ((missing.length > 1) ? "votes from ":"the vote from ") + missing.join(", ");
	var missing_text_2 = missing.join(", ") + " still missing";
    if (gs.votes[player_idx] == -1) {
		$(".candidate_text p").show().html("Here are the options, pick your choice");
	} else {
		$(".candidate_text p").show().html("Thanks. You can change your vote, " + missing_text_2);
	}
}

var result_html = function(gs, i, text) {
    var result_name = "True answer ";
    if (i < gs.n_players) result_name = gs.player_names[i] + "'s answer ";
    var html = '<div class="result_div"><span>' + result_name + "</span>";
    html += '<button type="button" class="btn btn-outline-dark candidate_button disabled">';
    html += text;
    html += "</button>"
    var guessed = [];
    for (var j=0; j < gs.n_players; j++) {
        if (gs.votes[j] == i) guessed.push(gs.player_names[j]);
    }
    if (guessed.length > 0) {
        html += "<span>";
        html += " voted by " + guessed.join(", ");
        html += "</span>";
    }
    html +="</div>";
    return html;
}

var candidate_html = function(text) {
    var html = '<div class="button_div"><button type="button" class="btn btn-outline-dark candidate_button">'
    html = html + text;
    html = html + '</button></div>';
    return html;
}


var display = function(session, gs, player_name) {
	$("div.manage_container").hide();
	$("div.game_container").show();
	var player_idx = gs.player_names.indexOf(player_name);
	if (player_idx < 0) {
		console.error("Couldn't find player name " + player_name + " among names " + gs.player_names);
	}
	if (debug) {
		console.log("Deck index: " + gs.deck_index);
		console.log("Candidates: " + gs.candidates);
	}
	
    q = questions[gs.deck_order[gs.deck_index]];
    var converter = new showdown.Converter();
    html_question = converter.makeHtml(q.question);
    html_question = html_question.replace("<BLANK>", "___________");
    $("p.question_caption").html(html_question);

    // Answer form
    if(gs.stage == 0) {
        $("div.answer").show();
		var send_answer = function() {
            console.log("Send answer called!");
            $(".answer_form input.the_answer").off("focusout");
            $(".answer_form").off("submit").hide();
            var a = $(".answer_form input").val().trim();
            if (a != "") {
                pub_command(session, {
                    name: "propose",
                    arg_1: player_idx,
                    arg_2: $(".answer_form input").val()
                });
            }
        }
        var missing = [];
        for (var i = 0; i<gs.n_players; i++) {
            if (gs.candidates[i] == "") missing.push(gs.player_names[i]);
        }
        var missing_text = "Waiting for " + missing.join(", ");
        if (gs.candidates[player_idx] == "") {
            $("p.answer_caption").show().text("Your answer, " + player_name);
            $(".answer_form").show().off("submit").on("submit", function(event){
                console.log("Calling send answer from submit!");
                event.preventDefault();
                send_answer();
            });
            $(".answer_form input.the_answer").off("focusout").on("focusout", function(event){
                console.log("Calling send answer from focus!");
                send_answer();
            });
        } else {
            $("p.answer_caption").show().text("Thanks. " + missing_text);
            $(".answer_form input").val("");
            $(".answer_form input.the_answer").off("focusout");
            $(".answer_form").off("submit").hide();
        }
	} else {
        $("div.answer").hide();
        $(".answer_form input").val("");
        $(".answer_form input.the_answer").off("focusout");
        $(".answer_form").off("submit").hide();
	}
	
    // Voting
    if(gs.stage == 0) {
        $("div.candidates").hide();
    } else if (gs.stage == 1) {
        $("div.candidates").show();
        var candidate_order = shuffle([...Array(gs.n_players + 1).keys()]);
        $("div.candidate_buttons").empty();
        display_missing_votes(gs, player_name);
        for (var i = 0; i <= gs.n_players; i++) {
            if (candidate_order[i] == gs.n_players) {
                text = q.answer;
            } else {
                text = gs.candidates[candidate_order[i]];
            }
            $("div.candidate_buttons").append(candidate_html(text));
        }
        var vote = function(idx) {
            if(candidate_order[idx] == player_idx) {
                return;
            }
            $(".candidate_button").removeClass("active");
            $(".candidate_button").eq(idx).addClass("active");
            var real_vote = candidate_order[idx];
            var votes_so_far = 0;
            for (var i=0; i<gs.n_players; i++) {
                // use the_gs so it is up-to-date even without refresh
                if (the_gs.votes[i] != -1) votes_so_far ++;
            }
            pub_command(session, {
                name: "vote",
                arg_1: player_idx,
                arg_2: real_vote,
            });
            if (votes_so_far < gs.n_players - 1) display_missing_votes(gs, player_name);
        }
        if (gs.votes[player_idx] != -1) {
            $(".candidate_button").eq(candidate_order.indexOf(gs.votes[player_idx])).addClass("active");
        }
        $(".candidate_button").each(function(idx) {
            if(candidate_order[idx] == player_idx) {
                $(".candidate_button").eq(idx).addClass("disabled");
            } else {
                $(this).click(() => vote(idx));
            }
        })
    } else { // stage 2
        $("div.candidates").show();
        $(".candidate_text p").hide();
        $("div.candidate_buttons").empty();
        for (var i = 0; i <= gs.n_players; i++) {
            if (i == gs.n_players) {
                text = q.answer;
            } else {
                text = gs.candidates[i];
            }
            $("div.candidate_buttons").append(result_html(gs, i, text));
        }
    }
        
	// // Candidate player names
	// if (gs.stage < 3) {
	// 	$(".candidates p").hide().text("");
	// } else {
	// 	for (var i=0; i<gs.candidates_shown.length; i++) {
	// 		var idx = gs.candidates.indexOf(gs.candidates_shown[i]);
	// 		var bg_color = (idx == gs.turn) ? "yellow":"white";
	// 		$(".candidates p.proposed").eq(i).show()
	// 			.css("background-color", bg_color)
	// 			.text(gs.player_names[idx]);
	// 	}
	// 	for (var i=gs.candidates.length; i<$(".candidates p.proposed").length; i++) {
	// 		$(".candidates p.proposed").eq(i).hide().text("");
	// 	}
	// 	var voted_whom = [];
	// 	for (var i=0; i < gs.n_players; i++) {
	// 		if (i != gs.turn) voted_whom.push(gs.candidates_shown.indexOf(gs.votes[i]));
	// 		else voted_whom.push(-1);
	// 	}
	// 	display_voted(gs, voted_whom);
	// 	$(window).resize(() => display_voted(gs, voted_whom));
	// }

	// // Candidates
	// if (gs.stage <= 1) {
	// 	$(".candidates img").hide().attr("src", "");
	// }
	// if (gs.stage == 0) {
	// 	$(".candidate_text p").hide();
	// } else if (gs.stage == 1) {
	// 	display_missing_candidates(gs, player_name);
	// } else if (gs.stage >= 2) {
	// 	set_candidates_size(gs, player_idx);
	// 	$(window).resize(() => set_candidates_size(gs, player_idx));
	// 	for (var i=gs.candidates.length; i<$(".candidates img").length; i++) {
	// 		$(".candidates img").eq(i).hide().attr("src", "");
	// 	}	
	// 	$(".candidates img").off('click');

	// 	if (gs.stage == 2) {
	// 		display_missing_votes(gs, player_name);
	// 		$(".candidates img").on('click',function(){
	// 			var idx_in_proposals = $(".candidates img").index($(this));
	// 			var card_id = gs.candidates_shown[idx_in_proposals];
	// 			if (card_id == gs.candidates[player_idx]) {
	// 				$(this).animate({opacity: 0.5}, 200);
	// 				$(this).animate({opacity: 1}, 200);
	// 				return;
	// 			}
	// 			$(".zoom img").attr("src", "img/Img_"+(card_id+1)+".jpg").show();
	// 			var ptext = $(".zoom p");
	// 			if (player_idx != gs.turn) {
	// 				var vote_this = function() {
	// 					$(".candidates img").css("background-color", "");
	// 					$(".candidates img").eq(idx_in_proposals)
	// 						.css("background-color", "red");
	// 					ptext.stop().hide();
	// 					$("div.zoom").hide();
	// 					var votes_so_far = 0;
	// 					for (var i=0; i<gs.n_players; i++) {
	// 						// use the_gs so it is up-to-date even without refresh
	// 						if (the_gs.votes[i] != -1) votes_so_far ++;
	// 					}
	// 					console.log("Votes so far: " + votes_so_far, " the_gs.votes: " + the_gs.votes);
	// 					pub_command(session, {
	// 						name: "vote",
	// 						arg_1: player_idx,
	// 						arg_2: card_id,
	// 					});
	// 					if (votes_so_far < gs.n_players - 2) display_missing_votes(gs, player_name);
	// 				}
	// 				ptext.text("Votar esta")
	// 					.off("click")
	// 					.on("click", vote_this)
	// 					.show();
	// 				$(".zoom img").off("click").on("click", vote_this);
	// 				function startAnimation() {
	// 					ptext.animate({opacity: 0}, 3000);
	// 					ptext.animate({opacity: 1}, 3000, startAnimation);
	// 				}
	// 				startAnimation();
	// 			} else {  // let the mano just see the zoomed card
	// 				$(".zoom img").off("click").on("click", () => {$(".zoom img").hide()});
	// 				ptext.off("click").hide();
	// 			}
	// 			$("div.zoom").css("top", "2%").height("50%").show();
	// 		});
	// 	} else {  // stage 4
	// 		$(".candidate_text p").show().text("Resultados:");
	// 	}
	// }
	
	// Scoreboard
	for (i=0; i<gs.n_players; i++) {
		color = (gs.scores[i] >= 5000) ? "red":"black";
		$(".player_names td").eq(i).text(gs.player_names[i]).css("color", color);
		$(".player_scores td").eq(i).text(gs.scores[i]).css("color", color);
	}
	
	// Next round
	if (gs.stage < 2 || !is_chief) {
		$(".next_round button").hide();
	} else {
		$(".next_round button").show()
		.on("click", function() {
			$(".next_round button").off("click").hide();
			pub_command(session, {"name": "next_round"});
		});
	}

}

var session_promise = diffusion.connect(diffusion_params);
session_promise.catch(error => {alert("Error setting up Diffusion session:" + error)});
$.getJSON("questions.json")
    .fail(() => {throw("Failed to load questions file!")})
    .done(json => {
        questions = json.normal;
        final_questions = json.final;
        console.log("Loaded " + questions.length + " normal questions and " +
                    final_questions.length + " final questions");
        if (is_chief) {
            if (!getCookie("game_is_on")) {
                $("div.game_container").hide();
                $("div.manage_container").show();
                if (debug) { console.log("Setting up game.");}
                session_promise.then(setup_game).then(setup_player_name);
            } else {
                console.log("game_is_on value " + getCookie("game_is_on"));		
                if (debug) { console.log("Game is on.");}
                session_promise.then(chief_start_game);
            }
        } else {
            $("div.manage").hide();
            if (debug) { console.log("Connecting to game.");}
            session_promise.then(setup_player_name);
        }
    })