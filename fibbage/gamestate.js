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
	this.stage = 0;  // 0 - Players write answers; 1 - Players pick answers; 2 - Results
    this.deck_index = 0;
    this.misleading_proposal_idx = -1;
	this.candidates = [];
	this.votes = [];
	this.scores = [];
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
	
    this.init = function() {
		if (debug) {
			console.log("Initializing game state");
		}
		// shuffle(this.deck_order);
        this.deck_index = -1;
        this.next_round();

        // Clear scores
        this.scores.length = 0;
		for (var i = 1; i <= this.n_players; i++) {
			this.scores.push(0);
		}
		return true;  // always refresh
	}
	
	this.next_round = function() {
        this.stage = 0;
		// Clear state
		this.candidates.length = 0;
		this.votes.length = 0;
		for (var i = 1; i <= this.n_players; i++) {
			this.candidates.push("");
			this.votes.push(-1);
		}
        this.deck_index = 0;  // += 1
        this.misleading_proposal_idx = Math.floor(Math.random() * 999999);
		return true;  // always refresh
	}
	
	this.propose = function(player_idx, proposal) {
		if(debug) {
			console.log("Proposing, player_idx=" + player_idx + ", proposal=" + proposal);
		}
		if (this.stage > 0) {
			throw "Got a proposal in stage " + this.stage;
		}

		this.candidates[player_idx] = proposal;

		if (this.candidates.every( v => v != "" )) {  // everyone has proposed
			if(debug) console.log("All players have proposed");
			this.stage = 1;
			return true;
		}
		return true;  // always refresh, it's not a problem for the candidate phase
	}

	this.vote = function(player_idx, vote_idx) {
		if(this.stage != 1) { // wrong stage
			throw "Got a vote in stage " + this.stage;
		}
		if(vote_idx < 0 || vote_idx > this.n_players) {
			throw ("Vote idx " + vote_idx + " out of bounds for " +
				   this.n_players + " players")
		}
		this.votes[player_idx] = vote_idx;
		var n_votes = 0;
		for (var i = 0; i < this.n_players; i++) {
			if (this.votes[i] > this.n_players || this.votes[i] < -1) {
                throw ("Illegal vote! Votes are " + this.votes)
            } else if (this.votes[i] >=0) {
                n_votes ++;
            }
		}
		if (n_votes != this.n_players) return false;
		
		// Update scores
		for (var i = 0; i < this.n_players; i++) {
			if (this.votes[i] == this.n_players) {
				this.scores[i] += 1000;  // player guessed correctly
			} else {
				this.scores[this.votes[i]] += 500;  // player voted for other player
			}
		}
		this.stage = 2;  // show results
		return true;
	}
}
