var debug = true;
// var is_chief = (navigator.userAgent.indexOf("Chrome") == -1) && (navigator.userAgent.indexOf("Firefox") == -1);
var is_chief = window.location.search.substring(1).indexOf("chief") >= 0;
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
	if (obj["name"] == "choose_topic") {
		refresh = gs.choose_topic(obj["arg_1"]);
	}
	if (obj["name"] == "skip_question") {
		refresh = gs.skip_question();
	}
	if (obj["name"] == "propose") {
		refresh = gs.propose(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "vote") {
		refresh = gs.vote(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "reward") {
		refresh = gs.reward(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "unreward") {
		refresh = gs.unreward(obj["arg_1"], obj["arg_2"]);
	}
	if (obj["name"] == "next_round") {
		refresh = gs.next_round();
	}
	gs.refresh = refresh;
	session.topics.updateValue('fibbage/gamestate',
							   JSON.stringify(gs),
							   diffusion.datatypes.json());
}

var check_answer = function(a, q, player_idx, gs) {
    a = a.trim().toLowerCase();
    if (a == "") return false;
    for (var i=0; i<gs.n_players; i++) {
        if (a == gs.candidates[i].trim().toLowerCase()) {
            $('#takenAnswerModal').modal();
            return false;
        }
    }
    if (a == q.answer.trim().toLowerCase()) {
        $('#takenAnswerModal').modal();
        return false;
    }
    for (var i=0; i<q.alternateSpellings.length; i++) {
        if (a == q.alternateSpellings[i].trim().toLowerCase()) {
            $('#takenAnswerModal').modal();
            return false;
        }
    }
    return true;
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
		$("p.candidate_text").show().text("Here are the options, take your pick");
	} else {
		$("p.candidate_text").show().text("Thanks. You can change your vote, " + missing_text_2);
	}
}


var final_score_html = function(gs, i, winner) {
    var result_name = "True answer ";
    if (i < gs.n_players) result_name = gs.player_names[i] + "'s answer ";
    if (i > gs.n_players) result_name = "Computer's answer ";
    var html = '<div class="col-12 my-1">'
    html += '<div class="row content-justify-start">'
    html += '<div class="col-3 my-1">'
    html += '<span class="h3">' + gs.player_names[i] + "</span>";
    html +="</div>";
    html += '<div class="col-3 my-1">'
    html += '<button type="button" class="btn btn-outline-dark candidate_button disabled">';
    html += gs.scores[i];
    html += "</button>"
    html +="</div>";
    html += '<div class="col-3 my-1">'
    if (winner) {
        html += '<span><i class="winner_icon fa fa-trophy fa-2x"></i></span>';
    }
    html +="</div>";
    html += '<div class="col-3 my-1">'
    html += '<span><i class="winner_icon fa fa-smile-o fa-1x"></i></span>';
    html += '<button type="button" class="btn btn-outline-dark candidate_button disabled">';
    html += gs.reward_scores[i];
    html += "</button>"
    html +="</div>";
    html +="</div>";
    html +="</div>";
    return html;
}

var result_html = function(gs, i, text) {
    var result_name = "True answer ";
    if (i < gs.n_players) result_name = gs.player_names[i] + "'s answer ";
    if (i > gs.n_players) result_name = "Computer's answer ";
    var html = '<div class="result_div col-12 my-1">'
    html += '<span>' + result_name + "</span>";
    if (i == gs.n_players) {
        html += '<button type="button" class="btn btn-primary candidate_button disabled">';
    } else {
        html += '<button type="button" class="btn btn-outline-dark candidate_button disabled">';
    }
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
    } else {
        html += "<span>";
        html += " not voted"
        html += "</span>";
    }
    html +="</div>";
    return html;
}

var topic_html = function(topic, active) {
    var html = '<div class="col-6 col-sm-4 col-md-3 my-1" style="text-align:center">'
    html += '<div class="button_with_wrapping_text">'
    if (active) {
        html += '<button type="button" class="btn m-0 btn-lg btn-outline-dark topic_button">'
    } else {
        html += '<button type="button" class="btn m-0 btn-lg btn-outline-dark topic_button" disabled>'
    }
    html += topic;
    html += '</button>';
    html += '</div>';
    html += '</div>';
    return html;
}

var candidate_html = function(text, active_reward) {
    var html = '<div class="col-6 col-sm-4 col-md-3 my-1" style="text-align:center">'
    html += '<div class="button_with_wrapping_text">'
    html += '<button type="button" class="btn m-0 btn-lg btn-outline-dark candidate_button">'
    html += text;
    html += '</button>';
    if (active_reward) {
        html += '<button type="button" class="reward_button btn btn-sm btn-warning m-1 p-0 active" data-toggle="button" aria-pressed="true" autocomplete="off">';
    } else {
        html += '<button type="button" class="reward_button btn btn-sm btn-warning m-1 p-0" data-toggle="button" aria-pressed="false" autocomplete="off">';
    }
    html += '<i class="reward_icon fa fa-smile-o fa-1x"></i>'
    html += '</button>';
    html += '</div>';
    html += '</div>';
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
	
    // Last round
    if(gs.round == gs.n_rounds - 1 && gs.stage != 4) {
        $("div.last_round").show();
    } else {
        $("div.last_round").hide();
    }
    
    // Topics
    if(gs.stage == 0) {
        if (gs.turn == player_idx) $("p.topic_text").text("Choose a topic, " + player_name);
        else $("p.topic_text").text("Waiting for " + gs.player_names[gs.turn] + " to choose a topic among:");
        $("div.topic_buttons").empty();
        if (gs.round == gs.n_rounds - 1) {
            qs = final_questions;
        } else {
            qs = questions;
        }
        for (var i = 0; i < gs.topic_idx.length; i++) {
            var topic = qs[gs.topic_idx[i]].category;
            $("div.topic_buttons").append(topic_html(topic, gs.turn == player_idx));
        }
        var choose_topic = function(button_idx) {
            var topic_idx = gs.topic_idx[button_idx];
            pub_command(session, {
                name: "choose_topic",
                arg_1: topic_idx,
            });
        }
        $(".topic_button").each(function(button_idx) {
            $(this).click(() => choose_topic(button_idx));
        });
        $("div.topics").show();
    } else {
        $("div.topics").hide();
    }
    
    // Question
    if(gs.stage == 0 || gs.stage == 4) {
        $("div.question").hide();
    } else {
        if (gs.round == gs.n_rounds - 1) {
            var q = final_questions[gs.deck_final_order[gs.deck_final_index]];
        } else {
            var q = questions[gs.deck_order[gs.deck_index]];
        }
        var converter = new showdown.Converter();
        html_question = converter.makeHtml(q.question);
        html_question = html_question.replace("<BLANK>", "___________");
        $("div.question").html(html_question);
        $("div.question").show();
    }

    // Answer form
    if(gs.stage == 1) {
        $("div.answer").show();
        if (is_chief) {
            $(".skip_question button").show()
                .off("click")
                .on("click", function() {
                    $(".skip_question button").off("click").hide();
                    pub_command(session, {"name": "skip_question"});
                });
        } else {
            $(".skip_question button").hide();
        }
		var send_answer = function() {
            var a = $(".answer_form input").val().trim();
            if (check_answer(a, q, player_idx, gs)) {
                $(".answer_form input.the_answer").off("focusout");
                $(".answer_form").off("submit").hide();
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
    if(gs.stage <= 1 || gs.stage == 4) {
        $("div.candidates").hide();
    } else if (gs.stage == 2) {
        var n_proposals = gs.n_players + 1;
        if (q.suggestions.length > 0) n_proposals += 1;
        $("div.candidates").show();
        var candidate_order = [...Array(n_proposals).keys()];
        candidate_order.splice(player_idx, 1);
        shuffle(candidate_order);
        $("div.candidate_buttons").empty();
        display_missing_votes(gs, player_name);
        for (var i = 0; i < candidate_order.length; i++) {
            if (candidate_order[i] == gs.n_players + 1) {
                text = q.suggestions[gs.misleading_proposal_idx % q.suggestions.length];
            } else if (candidate_order[i] == gs.n_players) {
                text = q.answer;
            } else {
                text = gs.candidates[candidate_order[i]];
            }
            var active_reward = (gs.rewards[player_idx].indexOf(candidate_order[i]) != -1);
            $("div.candidate_buttons").append(candidate_html(text, active_reward));
        }
        var vote = function(idx) {
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
            if (votes_so_far < gs.n_players) display_missing_votes(gs, player_name);
        }
        var reward = function(idx) {
            var action = "reward"
            if ($(".reward_button").eq(idx).hasClass("active")) {
                action = "unreward";
            }
            console.log("Action is " + action)
            var real_vote = candidate_order[idx];
            pub_command(session, {
                name: action,
                arg_1: player_idx,
                arg_2: real_vote,
            });
        }
        if (gs.votes[player_idx] != -1) {
            $(".candidate_button").eq(candidate_order.indexOf(gs.votes[player_idx])).addClass("active");
        }
        $(".candidate_button").each(function(idx) {
            $(this).click(() => vote(idx));
        })
        $(".reward_button").each(function(idx) {
            $(this).click(() => reward(idx));
        })
    } else { // stage 3
        var n_proposals = gs.n_players + 1;
        if (q.suggestions.length > 0) n_proposals += 1;
        $("div.candidates").show();
        $("p.candidate_text").hide();
        $("div.candidate_buttons").empty();
        for (var i = 0; i < n_proposals; i++) {
            if (i == gs.n_players + 1) {
                text = q.suggestions[gs.misleading_proposal_idx % q.suggestions.length];
            } else if (i == gs.n_players) {
                text = q.answer;
            } else {
                text = gs.candidates[i];
            }
            $("div.candidate_buttons").append(result_html(gs, i, text));
        }
    }
	
	// Scoreboard
	if (gs.stage < 4) {
        for (i=0; i<gs.n_players; i++) {
            color = ((gs.scores[i] > 0) && (gs.scores[i] == Math.max(...gs.scores))) ? "red":"black";
            $(".player_names div").eq(i).show().text(gs.player_names[i]).css("color", color);
            $(".player_scores div").eq(i).show().text(gs.scores[i]).css("color", color);
        }
        for (i=gs.n_players; i<12; i++) {
            $(".player_names div").eq(i).hide();
            $(".player_scores div").eq(i).hide();
        }
		$("div.scoreboard").show();
    } else {
		$("div.scoreboard").hide();
    }
	
	// Next round
	if (gs.stage != 3 || !is_chief) {
		$("div.next_round").hide();
	} else {
        var text = (gs.round == gs.n_rounds - 1) ? "Next round":"Final score"
        $(".next_round button").show()
        .text(text)
        .off("click")
		.on("click", function() {
			$(".next_round button").off("click").hide();
			pub_command(session, {"name": "next_round"});
		});
        $("div.next_round").show()
	}

    // Final scores
	if (gs.stage < 4) {
        $("div.final_scores").hide();
    } else {
        $("div.final_score_buttons").empty();
        var player_order = [...Array(gs.n_players).keys()];
        player_order.sort(function(i,j) {return(gs.scores[j] - gs.scores[i])});
        console.log("player_order " + player_order)
        for (var i = 0; i < player_order.length; i++) {
            var winner = (gs.scores[player_order[i]] == gs.scores[player_order[0]]);
            $("div.final_score_buttons").append(final_score_html(gs, player_order[i], winner));
        }
        if (is_chief) {
            $(".new_game button").show()
                .off("click")
                .on("click", function() {
                    $(".new_game button").off("click").hide();
                    pub_command(session, {"name": "init"});
                });
        } else {
            $(".new_game button").hide();
        }
        $("div.final_scores").show();
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