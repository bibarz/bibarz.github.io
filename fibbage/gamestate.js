const allEqual = arr => arr.every( v => v === arr[0] );

var shuffle = function (array) {

	var currentIndex = array.length;
	var temporaryValue, randomIndex;

	// While there remain elements to shuffle...
	while (0 !== currentIndex) {
		// Pick a remaining element...
		randomIndex = Math.floor(Math.random() * currentIndex);
		currentIndex -= 1;

		// And swap it with the current element.
		temporaryValue = array[currentIndex];
		array[currentIndex] = array[randomIndex];
		array[randomIndex] = temporaryValue;
	}

	return array;

};

var GameState = function(n_questions, n_final_questions, player_names) {
	this.n_players = player_names.length;
	this.player_names = player_names.slice();
	this.n_questions = n_questions;
	this.n_final_questions = n_final_questions;
    this.deck_order = [...Array(this.n_questions).keys()];
    shuffle(this.deck_order);
    this.deck_index = -1;
    this.deck_final_order = [...Array(this.n_final_questions).keys()];
    this.deck_final_index = -1;
    this.topic_idx = [];
    this.turn = 0;
    this.n_rounds = 8;
    this.n_topics = 8;  // (max) topics per round
	this.round = 0;
	this.stage = 0;  // 0-Topic choice 1-Players write answers; 2-Players pick answers; 3-Results; 4-Final results
    this.misleading_proposal_idx = -1;
	this.candidates = [];
	this.votes = [];
	this.rewards = [];
	this.scores = [];
	this.reward_scores = [];
    // refresh is not really a state but a message field. When we do
	// a state update that does not require refreshing (e.g., a proposal),
	// we still must publish it, it is not enough that the chief keeps
	// track of the change, because if the chief reloads, it will get the
	// state from the topic and forget about the non-refresh state change.
	// So we always publish the new state, but the topic listener checks
	// the refresh field to decide whether to update the display or not.
	// (It is slightly tricky because even if refresh=False we need to
	// display when it is the first message after a reload).
	this.refresh = true;
	
	this.randomize_topics = function() {
        var topic_names = [];
        var topic_idx = []
        // I'm accessing the global JSON questions here, I didn't want to
        if (this.round == this.n_rounds - 1) {
            var n_q = this.n_final_questions;
            var qs = final_questions;
            var d_idx = this.deck_final_index;
        } else {
            var n_q = this.n_questions;
            var qs = questions;
            var d_idx = this.deck_index;
        }
        for (var i=0; i<1000; i++) {
            var idx = Math.floor(Math.random() * (n_q - d_idx)) + d_idx;
            var topic = qs[idx].category;
            if (topic_names.indexOf(topic) == -1) {
                topic_names.push(topic);
                topic_idx.push(idx);
                if (topic_idx.length == this.n_topics) break;
            }
        }
        return topic_idx;
    }

    this.init = function() {
		if (debug) {
			console.log("Initializing game state");
		}
        this.round = -1;
        this.turn = -1;
        this.next_round();

        // Clear scores
        this.scores.length = 0;
        this.reward_scores.length = 0;
		for (var i = 1; i <= this.n_players; i++) {
			this.scores.push(0);
			this.reward_scores.push(0);
		}
		return true;  // always refresh
	}
	
	this.next_round = function() {
        this.stage = 0;
		// Clear state
		this.candidates.length = 0;
		this.votes.length = 0;
		this.rewards.length = 0;
		for (var i = 1; i <= this.n_players; i++) {
			this.candidates.push("");
			this.votes.push(-1);
			this.rewards.push([]);
		}
        this.round += 1;
		this.turn = (this.turn + 1) % this.n_players;
        if (this.round == this.n_rounds - 1) {
            this.deck_final_index = (this.deck_final_index + 1) % this.n_final_questions;
        } else {
            this.deck_index = (this.deck_index + 1) % this.n_questions;
        }
        this.topic_idx = this.randomize_topics();
        this.misleading_proposal_idx = Math.floor(Math.random() * 999999);
		return true;  // always refresh
	}

    this.skip_question = function() {
        this.stage = 0;
		// Clear state
		this.candidates.length = 0;
		this.votes.length = 0;
		this.rewards.length = 0;
		for (var i = 1; i <= this.n_players; i++) {
			this.candidates.push("");
			this.votes.push(-1);
			this.rewards.push([]);
		}
        if (this.round == this.n_rounds - 1) {
            this.deck_final_index = (this.deck_final_index + 1) % this.n_final_questions;
        } else {
            this.deck_index = (this.deck_index + 1) % this.n_questions;
        }
        this.topic_idx = this.randomize_topics();
        this.misleading_proposal_idx = Math.floor(Math.random() * 999999);
		return true;  // always refresh
	}
	
	this.choose_topic = function(topic_idx) {
        if (this.round == this.n_rounds - 1) {
            var n_q = this.n_final_questions;
            var d_idx = this.deck_final_index;
            var d_o =  this.deck_final_order;
        } else {
            var n_q = this.n_questions;
            var d_idx = this.deck_index;
            var d_o =  this.deck_order;
        }
        if (topic_idx < 0 || topic_idx >= n_q) {
			throw ("Topic idx " + topic_idx + " out of bounds for " + n_q + " questions.");
        }
        if (debug) {
			console.log("Chose topic idx " + topic_idx + ", will switch with current deck idx " + d_idx);
		}
        var tmp = d_o[d_idx];
        d_o[d_idx] = topic_idx;
        d_o[topic_idx] = tmp;
        this.stage = 1;
        return true;
    }

	this.propose = function(player_idx, proposal) {
		if(debug) {
			console.log("Proposing, player_idx=" + player_idx + ", proposal=" + proposal);
		}
		if (this.stage != 1) {
			throw "Got a proposal in stage " + this.stage;
		}

		this.candidates[player_idx] = proposal;

		if (this.candidates.every( v => v != "" )) {  // everyone has proposed
			if(debug) console.log("All players have proposed");
			this.stage = 2;
			return true;
		}
		return true;  // always refresh, it's not a problem for the candidate phase
	}

	this.reward = function(player_idx, reward_idx) {
		if(this.stage != 2) { // wrong stage
			throw "Got a reward in stage " + this.stage;
		}
		if(reward_idx < 0 || reward_idx > this.n_players + 1) {
			throw ("Reward idx " + reward_idx + " out of bounds for " +
				   this.n_players + " players");
        }
        if (this.rewards[player_idx].indexOf(reward_idx) == -1) {
            this.rewards[player_idx].push(reward_idx);
        }
        return false;  // never refresh for reward accounting
    }

	this.unreward = function(player_idx, reward_idx) {
		if(this.stage != 2) { // wrong stage
			throw "Got unreward in stage " + this.stage;
		}
		if(reward_idx < 0 || reward_idx > this.n_players + 1) {
			throw ("Reward idx " + reward_idx + " out of bounds for " +
				   this.n_players + " players");
        }
        for(var i=0; i < this.rewards[player_idx].length; i++) {
            if (this.rewards[player_idx][i] == reward_idx) {
                this.rewards[player_idx].splice(i, 1);
                i--;
            }
        }
        return false;  // never refresh for reward accounting
    }

    this.vote = function(player_idx, vote_idx) {
		if(this.stage != 2) { // wrong stage
			throw "Got a vote in stage " + this.stage;
		}
		if(vote_idx < 0 || vote_idx > this.n_players + 1) {
			throw ("Vote idx " + vote_idx + " out of bounds for " +
				   this.n_players + " players")
		}
		this.votes[player_idx] = vote_idx;
		var n_votes = 0;
		for (var i = 0; i < this.n_players; i++) {
			if (this.votes[i] > this.n_players + 1 || this.votes[i] < -1) {
                throw ("Illegal vote! Votes are " + this.votes)
            } else if (this.votes[i] >=0) {
                n_votes ++;
            }
		}
		if (n_votes != this.n_players) return false;
		
        // Update scores
        var multiplier = 1;
        if (this.round == this.n_rounds - 1) multiplier = 3;
		for (var i = 0; i < this.n_players; i++) {
			if (this.votes[i] == this.n_players) {
				this.scores[i] += multiplier * 1000;  // player guessed correctly
			} else if (this.votes[i] < this.n_players) {
				this.scores[this.votes[i]] += multiplier * 500;  // player voted for other player
			}  // otherwise the vote went to the computer suggestion
		}
		// Update rewards
		for (var i = 0; i < this.n_players; i++) {
            for (var j = 0; j < this.rewards[i].length; j++) {
                var reward_idx = this.rewards[i][j];
                if (reward_idx < 0 || reward_idx > this.n_players + 1) {
                    throw ("Illegal reward! Rewards are " + this.rewards);
                } else if (reward_idx < this.n_players) {
                    this.reward_scores[reward_idx] += 1;
                }  // just ignore the reward if it's for the computer's suggestions
            }
        }

        if (this.round == this.n_rounds - 1) {
            this.stage = 4;  // show final results
        } else {
            this.stage = 3;  // show round results
        }
		return true;
	}
}
