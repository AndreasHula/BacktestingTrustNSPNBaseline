#include "global.h"
#include "memorypool.h"

// Multiround Trustgame Estimation by A.Hula 2017, exact tree calculation

//Set irritated investor as a global node
static True_node* Irritation_Investor = new True_node();

//Max function for 2 doubles. Could be replaced by standard max
double maxd(double no_one, double no_two)
{
	double max_d = ( no_one < no_two ? no_two:no_one);
	return max_d;
}

//------------------------------------------- 1. Set up utilities and basic (=level-1) behavioural model -------------------------------------

Matrix initialize_trustee_utility(int trustee_guilt, double trustee_risk_aversion) //utility of trustee action for all possible investor actions
{
	Matrix ut = Zero_matrix(noa, 5); //holder for the utility values of 5 possible investor actions and 5 conditional trustee responses
	for(Matrix::size_type money_invested = 0; money_invested < ut.size1(); ++money_invested) //loop over investor choices
	{
		for(Matrix::size_type response = 0; response < nor(money_invested); ++response) //loop over trustee choices
		{
			double money_kept = 5.0*static_cast<double>(max_amount) - 5.0*static_cast<double>(money_invested); //amount the investor retains
			double money_returned = static_cast<double>(rbf*5*money_invested*response)/6.0; //amount the trustee returns
			double total_profit_investor = money_kept + money_returned; //round payoff for the investor
			double believed_guilt = static_cast<double>(trustee_guilt); //input trustee guilt transformed to double, guilt is "int" otherwise for use as an array index
			double total_profit_trustee = static_cast<double>(rbf*5*money_invested) - money_returned; //round payoff for trustee
			ut(money_invested,response) = (trustee_risk_aversion*static_cast<double>(rbf*5*money_invested)- money_returned -  believed_guilt*(0.1*believed_guilt+0.3)  * maxd(total_profit_trustee - total_profit_investor, 0.0));
			//Fehr-Schmidt utility. Guilt is transformed according to a mapping of 0 -> 0, 1-> 0.4 and 2 -> 1. Corresponding to the 3 guilt settings inference is based on.
		}
	}

	return ut;	//trustee choice utility structure is returned
}

Matrix initialize_trustee_probabilities(Matrix const& ut, double temperature) //Calculated likelihood of trustee choices, for all possible investor actions
{
	Matrix trustee_probabilities = Zero_matrix(noa, 5); //5 investor actions, 5 potential trustee responses (1 if investor choice was "invest 0")

	for(Matrix::size_type money_invested = 0; money_invested < trustee_probabilities.size1(); ++money_invested)
	{
		double sum = 0.0;	//exponential utilities will be summed up, to form the denominator of a logistic softmax
		double max_val = -100.0; //maximum utility is recorded, to prevent certain numerical issues (eliminate much small exponentials)
		for(Matrix::size_type money_returned = 0; money_returned < nor(money_invested); ++money_returned)	//record max trustee utility conditional on investor action
		{
			max_val = (max_val < 1.0/temperature*ut(money_invested, money_returned) ? 1.0/temperature*ut(money_invested, money_returned):max_val);
		}
		for(Matrix::size_type money_returned = 0; money_returned < nor(money_invested); ++money_returned)
		{
			if(1.0/temperature*ut(money_invested, money_returned)-max_val > -20.0)	//if the utility is close enough to the max to be relevant in the softmax:
			{																		// include in denominator of logistic softmax
				sum += exp(1.0/temperature*ut(money_invested, money_returned));
			}
		}
		for(Matrix::size_type money_returned = 0; money_returned < nor(money_invested); ++money_returned)
		{
			if(1.0/temperature*ut(money_invested, money_returned)-max_val > -20.0)
			{
				trustee_probabilities(money_invested, money_returned) = exp(1.0/temperature*ut(money_invested, money_returned))/sum; //calculate choice probability for logistic softmax
			}
			else
			{
				trustee_probabilities(money_invested, money_returned) = 0.0; //if value is much smaller than max utility: set probability to 0
			}
		}
	}

	return trustee_probabilities;	//return trustee choice probabilities
}

std::array<std::array<double, noa>, nob> initialize_expected_trustee_payoff(std::array<Matrix, nob> const& trustee_choice_probabilities, std::array<Matrix, nob> const& ut_system )
{	//precalculate trustee expected value for all possible investor actions and all possible trustee guilts (provides computational short-cut in the code)
	std::array<std::array<double,noa>, nob> Repayment_payoff; //holder for expected value of investor action, from the trustee perspective, given trustee guilt
	for(int t_guilt=0; t_guilt < nob; ++t_guilt)
	{
		for(int i_action = 0; i_action < noa; ++i_action)
		{
			Repayment_payoff[t_guilt][i_action] = 0.0; //initialize for subsequent summation
			for(int t_return = 0; t_return < nor(i_action); ++t_return) //includes own choice probability for each guilt
			{
				if(trustee_choice_probabilities[t_guilt](i_action, t_return) > 0.001) //numerical safeguard
				{
					Repayment_payoff[t_guilt][i_action] += ut_system[t_guilt](i_action, t_return)*trustee_choice_probabilities[t_guilt](i_action, t_return);
				}
			}
		}
	}

	return Repayment_payoff;	//return expected value of investor choice, from trustee perspective, given trustee guilt
}

std::array<std::array<std::array<double, noa>, nob>, nob> initialize_expected_outcome_utility(std::array<Matrix, nob> const& trustee_choice_probabilities, std::array<Matrix, nob> const& ui_system )
{	//precalculate the expected value of investment from the point of view of the investor, conditional on investor guilt and trustee guilt (for computational savings)
	std::array<std::array<std::array<double,noa>, nob>, nob> Repayment_expectation; //holder for value of investment expectation
	for(int i_guilt=0; i_guilt < nob; ++i_guilt)	//investor guilts
	{
		for(int t_guilt=0; t_guilt < nob; ++t_guilt)	//trustee guilts
		{
			for(int i_action = 0; i_action < noa; ++i_action) //possible investor actions
			{
				Repayment_expectation[i_guilt][t_guilt][i_action] = 0.0; //initialize for subsequent summation
				for(int t_return = 0; t_return < nor(i_action); ++t_return) //excludes own choice probability
				{
					if(trustee_choice_probabilities[t_guilt](i_action, t_return) > 0.001) //numerical safeguard
					{
						Repayment_expectation[i_guilt][t_guilt][i_action] += ui_system[i_guilt](i_action, t_return)*trustee_choice_probabilities[t_guilt](i_action, t_return);
					}
				}
			}
		}
	}

	return Repayment_expectation; //return expected value of investor choice, from investor perspective
}


//int min and max. Could be replaced by std
int mini(int no_one, int no_two)
{
	int min = ( no_one < no_two ? no_one:no_two);
	return min;
}

int maxa(int no_one, int no_two)
{
	int max = ( no_one < no_two ? no_two:no_one);
	return max;
}

Matrix initialize_ui_init(int guilt, double investor_risk_aversion)	//Calculate investor utility for a given investor guilt and all possible trustee responses
{
	Matrix ui_init = Zero_matrix(noa, 5);	//5 possible investor actions, 5 possible trustee responses (1 if the investor invested 0)
	for(Matrix::size_type money_invested = 0; money_invested < ui_init.size1(); ++money_invested)
	{
		for(Matrix::size_type response = 0; response < nor(money_invested); ++response)
		{
			double money_kept = 5.0*static_cast<double>(max_amount) - 5.0*static_cast<double>(money_invested);	//amount remaining with the investor at investment
			double money_returned = static_cast<double>(rbf*5*money_invested*response)/6.0;
			double total_profit_investor = money_kept + money_returned;
			double believed_guilt = static_cast<double>(guilt);
			double total_profit_trustee = static_cast<double>(rbf*5*money_invested) - money_returned;
			ui_init(money_invested,response) = (investor_risk_aversion*static_cast<double>(money_kept)+money_returned -  believed_guilt*(0.1*believed_guilt+0.3)  * maxd(total_profit_investor - total_profit_trustee, 0.0));
		}
	}

	return ui_init;
}

Matrix_vector initialize_utpi_init(double trustee_risk_aversion)	//trustee choice utility from the point of view of the investor
{
    Matrix_vector utpi_init(nob, Zero_matrix(noa, 5)); //utility of the trustee as perceived by the investor given each possible trustee guilt
	for(Matrix_vector::size_type b = 0; b < utpi_init.size(); ++b)
	{
		for(Matrix::size_type money_invested = 0; money_invested < utpi_init[b].size1(); ++money_invested)
		{
			for(Matrix::size_type response = 0; response < nor(money_invested); ++response)
			{
				double money_kept = 5.0*static_cast<double>(max_amount) - 5.0*static_cast<double>(money_invested);
				double money_returned = static_cast<double>(rbf*5*money_invested*response)/6.0;
				double total_profit_trustee = static_cast<double>(rbf*5*money_invested) - money_returned;
				double total_profit_investor =  money_kept+ money_returned;
				double believed_guilt = static_cast<double>(b);
				utpi_init[b](money_invested, response) = (trustee_risk_aversion*static_cast<double>(rbf*5*money_invested)-money_returned -  believed_guilt*(0.1*believed_guilt+0.3) * maxd(total_profit_trustee - total_profit_investor, 0.0));
			}
		}
	}

	return utpi_init;
}

void shift_updates(True_node** start_node, True_node** end_node, int const& level, double const& monetary_action)	//irritation shift calculation
{
	True_node* origin_node;	//pointer to hold the pointer to the agent node based on the preceeding history
	origin_node = *start_node;
	True_node* target_node;	//pointer to hold the pointer to the agent node based on the current history, which will contain updated shifts
	target_node = *end_node;
	std::array<std::array<double, noi>, noT+1> shifts = origin_node -> get_shifts(); //old shift state is obtained
	double expectation = 0.0;	//expectated value of the current interaction will need to be compared to the actual value of current monetary action

	if(level > 0)
	{
		expectation = origin_node-> get_expectation(level-1);	//levels are coded on by one level higher (i.e. expectation of level 0 actor is coded as 1), to prevent negative indices
	}
	else
	{
		expectation = origin_node-> get_expectation(0); //level -1s can have an expectation too. These are added as a special case here.
	}
	double disappointment =  expectation -  monetary_action;	//irritation can only occur if the monetary action is below the expected action
	for(int irr=0; irr < noi; ++irr)	//shift updates for all possible irritation values
	{
		shifts[level][irr] = (disappointment > 0.0 ? (1.0/static_cast<double>(noi-1))*static_cast<double>(irr):(-(1.0/static_cast<double>(noi-1))*static_cast<double>(irr))) + shifts[level][irr];
		//irritation occurs if disappointment is greater 0
		shifts[level][irr] = (shifts[level][irr] > 1.0 ? 1.0:shifts[level][irr]); //irritation is bounded at 1
		shifts[level][irr] = (shifts[level][irr] < 0.0 ? 0.0:shifts[level][irr]); //irritation is bounded at 0
		target_node -> set_shift( level, shifts[level][irr], irr);	//updated shift is carried forward into the "current history" node
	}


}

void belief_updates(True_node** start_node, True_node** end_node, int const& level, double const& monetary_action, int const& act, bool const& investor)
{
	True_node* origin_node;	//structure to keep the pointer to the already calculated node
	origin_node = *start_node;
	True_node* target_node;	//structure to keep the pointer to the to-be-calculated node
	target_node = *end_node;
	std::array<std::array<double,noi>, noT> Irr_beliefs = origin_node -> get_irr_beliefs(); //obtain irritation beliefs to be modified
	std::array<std::array<double, noi>, noT+1> shifts = origin_node -> get_shifts();	//obtain irritation shifts to be modified
	std::array<std::array<double,nob>, noT> beliefs = origin_node -> get_belief_parameters();	//obtain guilt beliefs to be modified
	std::array<std::array<double, noa>, nob> exp_payoffs = origin_node ->get_exp_payoffs(level-1);	//obtain choice likelihoods to be modified
	std::array<std::array<double, noa>, nob> i_exp_payoffs;	//obtain irritated choice likelihoods
	if(investor)
	{
		i_exp_payoffs = Irritation_Investor->get_exp_payoffs(0); //investor has a constant irritation behaviour
	}
	else
	{
		i_exp_payoffs = origin_node ->get_exp_payoffs(0);	//trustee irritation behaviour is conditional on investor choice
	}
	std::array<double, nob> running_belief_probability;	//holder for guilt belief to be modified
	std::array<double, noi> irr_probability;	//holder for irritation belief to be modified

	double sum =0.0;	//normalisation for guilt beliefs
	double isum =0.0;	//normalisation for irritation beliefs
	for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
	{
		running_belief_probability[guilt_counter] = beliefs[level-1][guilt_counter]; //set beliefs in holder
		sum += running_belief_probability[guilt_counter]; //summation
	}

	for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
	{
		running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;	//obtain approximate dirichlet multinomial probability for each guilt
	}
	for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
	{
		irr_probability[guilt_counter] = Irr_beliefs[level-1][guilt_counter];	//set irritation beliefs in holder
		isum += irr_probability[guilt_counter]; //summation
	}
	for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
	{
		irr_probability[guilt_counter] = irr_probability[guilt_counter]/isum;	//obtain approximate dirichlet multinomial probability for each guilt
	}

	for(int guilt=0; guilt < nob ; ++guilt)	//update for each guilt
	{
		//if(!(investor) && act==1 )
		//{
			for(int irr=0; irr < noi; ++irr)	//for each guilt, all irritation scenarios have to be considered
			{
				if( exp_payoffs[guilt][act] > 0.01 && irr_probability[irr] > 0.001)	//numerical safeguard
				{
					beliefs[level-1][guilt]+= (1.0-shifts[level-1][irr])*irr_probability[irr]*exp_payoffs[guilt][act]; //unshifted choice probability
				}
			}

			for(int irr=0; irr < noi; ++irr)
			{

				if( shifts[level-1][irr] > 0.001 && irr_probability[irr] > 0.001 && i_exp_payoffs[0][act] > 0.001 )	//numerical safeguard
				{
					beliefs[level-1][guilt]+= shifts[level-1][irr]*irr_probability[irr]*i_exp_payoffs[0][act]; //shifted (=irritated) choice probability
				}
			}
		//}
		target_node-> set_belief_parameters(guilt, beliefs[level-1][guilt], level-1);	//new 9approximate) belief priors are set in the new node
	}
	

	double tester = 0.0;	//variable to check if irritation actually happens
	for(int irr=0; irr < noi; ++irr)
	{
		tester = ( shifts[level-1][irr] > 0.0 ? 1.0:tester);
    }
	for(int irr=0; irr < 1; ++irr)
	{
 		if( tester>0.0 ) //update if there was a chance of irritation
		{

			for(int guilt=0; guilt < nob ; ++guilt)	//consider all possible guilt scenarios for the given irritation value
			{
				if(running_belief_probability[guilt]> 0.001 && shifts[level-1][irr] < 0.999 && exp_payoffs[guilt][act] > 0.001)
				{
					Irr_beliefs[level-1][irr] += (1.0-shifts[level-1][irr])*exp_payoffs[guilt][act]*running_belief_probability[guilt];
				}
			}

			for(int guilt=0; guilt < nob ; ++guilt)
			{
				if(running_belief_probability[guilt]> 0.001 && shifts[level-1][irr] > 0.001 && i_exp_payoffs[0][act] > 0.001)
				{
					Irr_beliefs[level-1][irr]  += shifts[level-1][irr]*i_exp_payoffs[0][act]*running_belief_probability[guilt];
				}
			}


            //target_node-> set_irr_parameters(irr, Irr_beliefs[level-1][irr], level-1);	//set new irritation beliefs

		}
    }

    tester = 0.0;

	for(int irr=1; irr < noi; ++irr)
	{
        tester = ( shifts[level-1][irr] > 0.0 ? 1.0:0.0);
		if( tester>0.0 )
		{

			for(int guilt=0; guilt < nob ; ++guilt)
			{
				if(running_belief_probability[guilt]> 0.001 && shifts[level-1][irr] < 0.999 && exp_payoffs[guilt][act] > 0.001)
				{
					Irr_beliefs[level-1][irr] += (1.0-shifts[level-1][irr])*exp_payoffs[guilt][act]*running_belief_probability[guilt];
				}
			}

			for(int guilt=0; guilt < nob ; ++guilt)
			{
				if(running_belief_probability[guilt]> 0.001 && shifts[level-1][irr] > 0.001 && i_exp_payoffs[0][act] > 0.001)
				{
					Irr_beliefs[level-1][irr]  += shifts[level-1][irr]*i_exp_payoffs[0][act]*running_belief_probability[guilt];
				}
			}

            

		}

	}
	for(int irr=0; irr < noi; ++irr)
	{	
		target_node-> set_irr_parameters(irr, Irr_beliefs[level-1][irr], level-1);
	}
}

void investor_expectation_calculation( int const& level, True_node** given_pointer, std::array<double, nob> const& running_belief_probability , std::array<double, noi> const& irr_probability)
{	//calculate the trustee expectation of the investor action
	//beliefs are included in the function as probabilities, NOT priors (though this would be possible)
	double expectation = 0.0; //holder for expectation
	True_node* node_pointer; //pointer to the to be modified node
	node_pointer = *given_pointer;
	std::array<std::array<double, noa>, nob> exp_payoffs;
	if(level > -1)
	{
		std::array<std::array<double, noi>, noT+1> shifts = node_pointer -> get_shifts(); //determine if investor acts under irritation
		if(level == 0) //special case for level 0 //technicality left from an earlier version
		{
			exp_payoffs = node_pointer -> get_exp_payoffs(0); //get non irritated payoffs
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr]>0.001)
				{
					if((1.0 - shifts[0][irr]) > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if(running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < noa ; ++i)
								{
									if(exp_payoffs[belief][i] > 0.0001)
									{
										expectation += (1.0-shifts[0][irr])*5.0*static_cast<double>(i)*exp_payoffs[belief][i]*
										running_belief_probability[belief]*irr_probability[irr]; //sum expectations over all non-irritated components, weighted by parameter likelihood
									}
								}
							}
						}
					}
				}
			}
			exp_payoffs = Irritation_Investor -> get_exp_payoffs(0);
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr] > 0.001)
				{
					if(shifts[0][irr] > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if(running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < noa ; ++i)
								{
									if(exp_payoffs[0][i] > 0.0001)
									{
										expectation += shifts[0][irr]*5.0*static_cast<double>(i)*exp_payoffs[0][i]*
										running_belief_probability[belief]*irr_probability[irr];	//sum expectations over all irritated components, weighted by parameter likelihood
									}
								}
							}
						}
					}
				}
			}

			node_pointer -> set_expectation(expectation, 0);		//set calculated expectation
		}
		else
		{
			exp_payoffs = node_pointer -> get_exp_payoffs(level); //case for a general level
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr] > 0.001)
				{
					if((1.0 -shifts[level][irr]) > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if(running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < noa ; ++i)
								{
									if(exp_payoffs[belief][i] > 0.0001)
									{
										expectation += (1.0-shifts[level][irr])*5.0*static_cast<double>(i)*exp_payoffs[belief][i]*
										running_belief_probability[belief]*irr_probability[irr];
									}
								}
							}
						}
					}
				}
			}
			exp_payoffs = Irritation_Investor -> get_exp_payoffs(0);
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr] > 0.001)
				{
					if(shifts[level][irr] > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if(running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < noa ; ++i)
								{
									if(exp_payoffs[0][i] > 0.0001)
									{
										expectation += shifts[level][irr]*5.0*static_cast<double>(i)*exp_payoffs[0][i]*
										running_belief_probability[belief]*irr_probability[irr];
									}
								}
							}
						}
					}
				}
			}

			node_pointer -> set_expectation(expectation, level);
		}
	}
	else
	{	//necessary special case for level -1 Trustee. Level -2 investor model includes no irritation. (Guilt belief could be removed here, since it is uniform always)
		exp_payoffs = node_pointer -> get_exp_payoffs(0);
		for(int belief=0; belief < nob ; ++belief)
		{
			if( running_belief_probability[belief] > 0.001)
			{
				for(int i=0; i < noa; ++i)
				{
					if(exp_payoffs[belief][i] > 0.0001)
					{
						expectation += 5.0*static_cast<double>(i)*exp_payoffs[belief][i]*
						running_belief_probability[belief];
					}
				}
			}
		}

		node_pointer -> set_expectation(expectation, 0);
	}

}

void trustee_expectation_calculation( int const& level, True_node** given_pointer, std::array<double, nob> const& running_belief_probability , std::array<double, noi> const& irr_probability, int const& i_act)
{	//calculation of investor expectation fo trustee action
	double expectation = 0.0;
	True_node* node_pointer;
	node_pointer = *given_pointer;
	std::array<std::array<double, noa>, nob> exp_payoffs;
	if(level > -1)
	{
		std::array<std::array<double, noi>, noT+1> shifts = node_pointer -> get_shifts();
		if(level == 0)
		{
			exp_payoffs = node_pointer -> get_exp_payoffs(0);
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr] > 0.001)
				{
					if((1.0 -shifts[0][irr]) > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if(running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < nor(i_act) ; ++i)
								{
									if(exp_payoffs[belief][i] > 0.0001)
									{
										expectation += (1.0-shifts[0][irr])*1.0/6.0*static_cast<double>(i*5*rbf*i_act)*exp_payoffs[belief][i]*
										running_belief_probability[belief]*irr_probability[irr];
									}
								}
							}
						}
					}
				}
			}
			exp_payoffs = node_pointer -> get_exp_payoffs(0);
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr] > 0.001)
				{
					if(shifts[0][irr] > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if(running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < nor(i_act) ; ++i)
								{
									if(exp_payoffs[0][i] > 0.0001)
									{
										expectation += shifts[0][irr]*1.0/6.0*static_cast<double>(i*5*rbf*i_act)*exp_payoffs[0][i]*
										running_belief_probability[belief]*irr_probability[irr];
									}
								}
							}
						}
					}
				}
			}

			node_pointer -> set_expectation(expectation, level);
		}
		else
		{
			exp_payoffs = node_pointer -> get_exp_payoffs(level);
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr] > 0.001)
				{
					if((1.0 -shifts[level][irr]) > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if( running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < nor(i_act) ; ++i)
								{
									if( exp_payoffs[belief][i] > 0.001)
									{
										expectation += (1.0-shifts[level][irr])*1.0/6.0*static_cast<double>(i*5*rbf*i_act)*exp_payoffs[belief][i]*
										running_belief_probability[belief]*irr_probability[irr];
									}
								}
							}
						}
					}
				}
			}
			exp_payoffs = node_pointer -> get_exp_payoffs(0);
			for(int irr=0; irr < noi; ++irr)
			{
				if(irr_probability[irr] > 0.001)
				{
					if(shifts[level][irr] > 0.001)
					{
						for(int belief=0; belief < nob ; ++belief)
						{
							if(running_belief_probability[belief] > 0.001)
							{
								for(int i=0; i < nor(i_act) ; ++i)
								{
									if(exp_payoffs[0][i] > 0.001)
									{
										expectation += shifts[level][irr]*1.0/6.0*static_cast<double>(i*5*rbf*i_act)*exp_payoffs[0][i]*
										running_belief_probability[belief]*irr_probability[irr];
									}
								}
							}
						}
					}
				}
			}

			node_pointer -> set_expectation(expectation, level);
		}
	}
	else
	{
		exp_payoffs = node_pointer -> get_exp_payoffs(0);
		for(int belief=0; belief < nob ; ++belief)
		{
			if( running_belief_probability[belief] > 0.001)
			{
				for(int i=0; i < nor(i_act) ; ++i)
				{
					if(exp_payoffs[belief][i] > 0.0001)
					{
						expectation += 1.0/6.0*static_cast<double>(i*5*rbf*i_act)*exp_payoffs[belief][i]*
						running_belief_probability[belief];
					}
				}
			}
		}
		node_pointer -> set_expectation(expectation, 0);
	}

}

void investor_k( int const& present_time
					 , int const& planning_horizon
					 , std::array<int,noT+1> guilt_parameter
					 , int const& k1
					 , std::array<int, global_time_horizon> const& pastactions
					 , std::array<int, global_time_horizon> const& responses
					 , std::array<Matrix, nob> const& ui_init_system
					 , std::array<Matrix, nob> const& ut_init_system
					 , std::array<std::array<double, noa>, nob> const& expected_trustee_payoff
					 , std::array<std::array<std::array<double, noa>, nob>, nob> const& expected_outcome_utlity
					 , double const& temperature
					, MEMORY_POOL<True_node>& mempool
					, True_node** reference_node);

void trustee_k(  int const& money_invested
					, int const& present_time
					, int const& planning_horizon
					, std::array<int,noT+1> guilt_parameter
					, int const& k2
					, std::array<int, global_time_horizon> const& pastactions
					, std::array<int, global_time_horizon> const& responses
					, std::array<Matrix, nob> const& ui_init_system
					, std::array<Matrix, nob> const& ut_init_system
					, std::array<std::array<double, noa>, nob> const& expected_trustee_payoff
					, std::array<std::array<std::array<double, noa>, nob>, nob> const& expected_outcome_utility
					, double const& temperature
					, MEMORY_POOL<True_node>& mempool
					,  True_node** reference_node
					)
{
	std::vector<double> trustee_val (noa*global_time_horizon,0.0);
	std::array<int, global_time_horizon> action;
	std::array<int, global_time_horizon> response;
	std::array<int, global_time_horizon> sorted_hist;
	int trustee_guilt;
	std::array<int, 1> initial_count;
	std::array<int, 1> count;

	for(int l=0; l < global_time_horizon; ++l)
	{
		action[l]=pastactions[l];
	}

	for(int l=0; l < global_time_horizon; ++l)
	{
		response[l]=responses[l];
	}

	True_node* current_node;
	current_node = *reference_node;
	True_node* Trustee_node_r;
	Trustee_node_r= *reference_node;
	True_node* root;
	root = *reference_node;

	int game = present_time;

	if( (k2 < 0) )
	{
		trustee_guilt = guilt_parameter[noT];

		int update = 0;
		double sum =0.0;
		double max_val = -100.0;
		for(int action_loc =0; action_loc < nor(money_invested); ++action_loc)
		{
			trustee_val[action_loc+noa*update] = ut_init_system[trustee_guilt](money_invested,action_loc);
			max_val = (max_val < trustee_val[action_loc+noa*update] ? trustee_val[action_loc+noa*update]:max_val);
			Trustee_node_r -> set_payoff(action_loc, trustee_val[action_loc+noa*update]);
		}
		for(int action_loc =0; action_loc < nor(money_invested); ++action_loc)
		{
			if(1.0/temperature*(trustee_val[action_loc+noa*update]-max_val) > -20.0)
			{
				sum += exp(1.0/temperature*trustee_val[action_loc + noa*update]);
			}
		}
		for(int action_loc =0; action_loc < nor(money_invested); ++action_loc)
		{
			if(1.0/temperature*(trustee_val[action_loc+noa*update]-max_val) > -20.0)
			{
				trustee_val[action_loc+noa*update] = exp(1.0/temperature*trustee_val[action_loc+noa*update])/sum;
			}
			else
			{
				trustee_val[action_loc + noa*update] = 0.0;
			}
			Trustee_node_r -> set_exp_payoffs(action_loc,trustee_val[action_loc+noa*update], trustee_guilt, 0);
		}



		Trustee_node_r ->confirm_exploration(trustee_guilt, 0);

	}
	else
	{

		if((planning_horizon- game) == 1)
		{
			int trustee_guilt = guilt_parameter[k2];
			int update = 0;
			//for(int update=0; update < (planning_horizon-present_time); ++update)
			//{
				double sum = 0.0;
				double max_val = -100.0;
				for(int action_loc =0; action_loc < nor(money_invested); ++action_loc)
				{
					trustee_val[action_loc+noa*update] = ut_init_system[trustee_guilt](money_invested, action_loc);
					Trustee_node_r -> set_payoff(action_loc, trustee_val[action_loc+noa*update]);
					max_val = (max_val < trustee_val[action_loc+noa*update] ? trustee_val[action_loc+noa*update]:max_val);

				}
				for(int action_loc =0; action_loc < nor(money_invested); ++action_loc)
				{
					if(1.0/temperature*(trustee_val[action_loc+noa*update]-max_val) > -20.0)
					{
						sum += exp(1.0/temperature*trustee_val[action_loc + noa*update]);
					}
				}
				for(int action_loc =0; action_loc < nor(money_invested); ++action_loc)
				{
					if(1.0/temperature*(trustee_val[action_loc+noa*update]-max_val) > -20.0)
					{
						trustee_val[action_loc+noa*update] = exp(1.0/temperature*trustee_val[action_loc+noa*update])/sum;
					}
					else
					{
						trustee_val[action_loc + noa*update] = 0.0;
					}
					Trustee_node_r -> set_exp_payoffs(action_loc,trustee_val[action_loc+noa*update], trustee_guilt, k2+1);
				}

			//}


			Trustee_node_r-> confirm_exploration(trustee_guilt, k2+1);
		}
		if((planning_horizon- game) == 2)
		{
			int trustee_guilt = guilt_parameter[k2];
			std::array<std::array<double, nob>, noT> beliefs;
			std::array<std::array<double, nob>, noT> updated_beliefs;
			std::array<double, noa> prob;
			std::array<double, nob> prob_guilt;
			std::array<std::array<double, noa>, nob> inv_exp_payoffs;
			std::array<std::array<double, noa>, nob> loc_payoffs;
			std::array<double, noa> t_util;
			std::array<double,nob> investor_belief_probability;
			std::array<double,nob> running_belief_probability;
			std::array<double,noi> investor_irr_probability;
			std::array<double,noi> running_irr_probability;
			std::array<std::array<double,noi>, noT> irr_beliefs;
			std::array<std::array<double, noi>, noT+1> local_shifts;
			beliefs = Trustee_node_r -> get_belief_parameters();
			irr_beliefs = Trustee_node_r-> get_irr_beliefs();
			double sum =0.0;
			for(int o = 0; o < nob; ++o)
			{
				sum += beliefs[k2][o];
			}
			for(int o = 0; o < nob; ++o)
			{
				investor_belief_probability[o] = beliefs[k2][o]/sum;
			}
			double isum = 0.0;
			for(int o = 0; o < noi; ++o)
			{
				isum += irr_beliefs[k2][o];
			}
			for(int o = 0; o < noi; ++o)
			{
				investor_irr_probability[o] = irr_beliefs[k2][o]/isum;
			}
			if(k2>0)
			{
				sum =0.0;
				isum = 0.0;
				for(int o = 0; o < nob; ++o)
				{
					sum += beliefs[k2-1][o];
				}
				for(int o = 0; o < noi; ++o)
				{
					isum += irr_beliefs[k2-1][o];
				}
				for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = beliefs[k2-1][guilt_counter]/sum;
				}
				for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
				{
					running_irr_probability[irr_counter] = irr_beliefs[k2-1][irr_counter]/isum;
				}
				trustee_expectation_calculation( k2-1, &Trustee_node_r, running_belief_probability , running_irr_probability, money_invested);
			}
			/*else
			{
				for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
				}
				for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
				{
					running_irr_probability[irr_counter] = 0.0;
				}
				running_irr_probability[0]= 1.0;
				trustee_expectation_calculation( -1, &Trustee_node_r, running_belief_probability , running_irr_probability, money_invested);
			}*/

			double b_sum =0.0;
			sum = 0.0;
			double max_val = -100.0;
			//for(int update=1; update < (planning_horizon-present_time); ++update)
			//{
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{

					t_util[first_ret] = ut_init_system[trustee_guilt](money_invested , first_ret);
					action[game] = money_invested;
					response[game] = first_ret;
					current_node = Trustee_node_r -> get_child(first_ret);
					if(!current_node)
					{
						current_node = new True_node();
						irr_beliefs = Trustee_node_r-> get_irr_beliefs();
						beliefs = Trustee_node_r -> get_belief_parameters();
						local_shifts = Trustee_node_r -> get_shifts();
						for(int l= 0; l < noT; ++l)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								current_node-> set_irr_parameters(irr, irr_beliefs[l][irr], l);
							}
							for(int b=0; b < nob; ++b)
							{
								current_node -> set_belief_parameters(b, beliefs[l][b], l);
							}
						}
						for(int level=0; level < noT+1; ++level)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								current_node -> set_shift(level, local_shifts[level][irr], irr);
							}
						}

						Trustee_node_r -> set_child(first_ret, current_node);
						
						//Initialize basic payoffs
						for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
						{
							guilt_parameter[noT]=guilt_counter;
							investor_k( game+1
							, planning_horizon
							, guilt_parameter
							, -1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &current_node);

						}

						//level -1
						if((k2+2)%2== 0)
						{
							double sum = 0.0;
							double isum = 0.0;
							beliefs = current_node -> get_belief_parameters();
							irr_beliefs = current_node -> get_irr_beliefs();
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[0][guilt_counter];
								sum += running_belief_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = irr_beliefs[0][guilt_counter];
								isum += running_irr_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
							}
							investor_expectation_calculation( 0, &current_node,  running_belief_probability , running_irr_probability);
						}
						else
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = 0.0;
							}
							running_irr_probability[0] = 1.0;
							investor_expectation_calculation( -1, &current_node,  running_belief_probability , running_irr_probability);
						}
					}

					irr_beliefs = Trustee_node_r-> get_irr_beliefs();
					beliefs = Trustee_node_r -> get_belief_parameters();
					local_shifts = Trustee_node_r -> get_shifts();
					for(int irr=0; irr < noi; ++irr)
					{
						current_node-> set_irr_parameters(irr, irr_beliefs[k2][irr], k2);
						if(k2>0)
						{
							current_node-> set_irr_parameters(irr, irr_beliefs[k2-1][irr], k2-1);
						}
					}
					for(int b=0; b < nob; ++b)
					{
						current_node -> set_belief_parameters(b, beliefs[k2][b], k2);
						if(k2>0)
						{
							current_node -> set_belief_parameters(b, beliefs[k2-1][b], k2-1);
						}
					}
					for(int irr=0; irr < noi; ++irr)
					{
						current_node -> set_shift(k2+1, local_shifts[k2+1][irr], irr);
						current_node -> set_shift(k2, local_shifts[k2][irr], irr);
					}
					double trustee_monetary_response = 1/6.0*static_cast<double>(rbf*5*money_invested*response[game]);
					if(!(money_invested==0))
					{
						if(k2>0 && money_invested>1 )
						{
							belief_updates(&Trustee_node_r, &current_node, k2, trustee_monetary_response , response[game], false);
						}
						shift_updates(&Trustee_node_r, &current_node, k2, trustee_monetary_response);
					}

					local_shifts = current_node -> get_shifts();
					for(int I_guilt=0; I_guilt < nob; ++I_guilt)
					{
						if(!(current_node->get_confirm_exploration(I_guilt, k2)))
						{
							if(k2>0)
							{
								guilt_parameter[k2-1]=I_guilt;
							}
							guilt_parameter[noT]=I_guilt;
							investor_k( game+1
							, planning_horizon
							, guilt_parameter
							, k2-1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &current_node);
							current_node->confirm_exploration(I_guilt, k2);
						}

					}
					if(!current_node -> expectation_set(k2))
					{
						if(k2>0)
						{
							double sum = 0.0;
							double isum = 0.0;
							beliefs = current_node -> get_belief_parameters();
							irr_beliefs = current_node -> get_irr_beliefs();
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[k2][guilt_counter];
								sum += running_belief_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = irr_beliefs[k2][guilt_counter];
								isum += running_irr_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
							}
							investor_expectation_calculation( k2, &current_node,  running_belief_probability , running_irr_probability);
						}
						/*else
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/nob;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = 0.0;
							}
							running_irr_probability[0] = 1.0;
							investor_expectation_calculation( -1, &current_node,  running_belief_probability , running_irr_probability);
						}*/
					}

					loc_payoffs = current_node-> get_exp_payoffs(k2);
					inv_exp_payoffs = Irritation_Investor -> get_exp_payoffs(0);
					local_shifts = current_node -> get_shifts();
					for(int irr=0; irr < noi; ++irr)
					{
						if(investor_irr_probability[irr]> 0.0001 )
						{
							for(int I_guilt=0; I_guilt < nob; ++I_guilt)
							{
								if(investor_belief_probability[I_guilt] > 0.001)
								{
									for(int second_act = 0; second_act < noa; ++second_act)
									{
										if(loc_payoffs[I_guilt][second_act]> 0.001 && (1.0-local_shifts[k2][irr]) > 0.001)
										{
											t_util[first_ret] += investor_belief_probability[I_guilt]*expected_trustee_payoff[trustee_guilt][second_act]*
											investor_irr_probability[irr]*(1.0-local_shifts[k2][irr])*loc_payoffs[I_guilt][second_act];
										}
										if(inv_exp_payoffs[0][second_act]> 0.001 && local_shifts[k2][irr] > 0.001)
										{
											t_util[first_ret] += investor_irr_probability[irr]*investor_belief_probability[I_guilt]*expected_trustee_payoff[trustee_guilt][second_act]*
											local_shifts[k2][irr]*inv_exp_payoffs[0][second_act];	//	expected_trustee_payoff correct?
										}
									}
								}
							}
						}
					}

					Trustee_node_r -> set_payoff(first_ret, t_util[first_ret]);
				}

				//exp payoffs calc
				sum = 0.0;
				double max_util = -100.0;
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{
					max_util = (1.0/temperature*t_util[first_ret] > max_util ? 1.0/temperature*t_util[first_ret]: max_util);
				}
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{
					if(1.0/temperature*t_util[first_ret]-max_util > -20)
					{
						sum += exp(1.0/temperature*t_util[first_ret]);
					}
				}
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{
					if(1.0/temperature*t_util[first_ret]-max_util > -20)
					{
						t_util[first_ret] = exp(1.0/temperature*t_util[first_ret])/sum;
					}
					else
					{
						t_util[first_ret] = 0.0;
					}
					Trustee_node_r -> set_exp_payoffs(first_ret, t_util[first_ret], trustee_guilt, k2+1);
				}
			//}

			Trustee_node_r-> confirm_exploration(trustee_guilt, k2+1);

		}

		if(planning_horizon - game > 2)
		{
			int trustee_guilt = guilt_parameter[k2];
			True_node* Trustee_holder;
			std::array<std::array<double, nob>, noT> beliefs;
			std::array<std::array<double, nob>, noT> updated_beliefs;
			std::array<double, noa> prob;
			std::array<double, nob> prob_guilt;
			std::array<std::array<double, noa>, nob> inv_exp_payoffs;
			std::array<std::array<double, noa>, nob> loc_payoffs;
			std::array<std::array<double, noa>, nob> tru_prob;
			std::array<double, noa> t_util;
			std::array<double,nob> investor_belief_probability;
			std::array<double,nob> running_belief_probability;
			std::array<double,noi> investor_irr_probability;
			std::array<double,noi> running_irr_probability;
			std::array<std::array<double,noi>, noT> irr_beliefs;
			std::array<std::array<double, noi>, noT+1> local_shifts;
			beliefs = Trustee_node_r -> get_belief_parameters();
			irr_beliefs = Trustee_node_r-> get_irr_beliefs();
			double investor_monetary_action=0.0;
			double sum =0.0;
			for(int o = 0; o < nob; ++o)
			{
				sum += beliefs[k2][o];
			}
			for(int o = 0; o < nob; ++o)
			{
				investor_belief_probability[o] = beliefs[k2][o]/sum;
			}
			double isum = 0.0;
			for(int o = 0; o < noi; ++o)
			{
				isum += irr_beliefs[k2][o];
			}
			for(int o = 0; o < noi; ++o)
			{
				investor_irr_probability[o] = irr_beliefs[k2][o]/isum;
			}
			if(k2>0)
			{
				sum =0.0;
				isum = 0.0;
				for(int o = 0; o < nob; ++o)
				{
					sum += beliefs[k2-1][o];
				}
				for(int o = 0; o < noi; ++o)
				{
					isum += irr_beliefs[k2-1][o];
				}
				for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = beliefs[k2-1][guilt_counter]/sum;
				}
				for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
				{
					running_irr_probability[irr_counter] = irr_beliefs[k2-1][irr_counter]/isum;
				}
				trustee_expectation_calculation( k2-1, &Trustee_node_r,  running_belief_probability , running_irr_probability, money_invested);
			}
			/*else
			{
				for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
				}
				for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
				{
					running_irr_probability[irr_counter] = 0.0;
				}
				running_irr_probability[0]= 1.0;
				trustee_expectation_calculation( -1, &Trustee_node_r,  running_belief_probability , running_irr_probability, money_invested);
			}*/
			double b_sum =0.0;
			sum = 0.0;
			double max_val = -100.0;
			//for(int update=planning_horizon-game-1; update < (planning_horizon-present_time); ++update)
			//{
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{
					t_util[first_ret] = ut_init_system[trustee_guilt](money_invested , first_ret);
					action[game] = money_invested;
					response[game] = first_ret;
					current_node = Trustee_node_r -> get_child(first_ret);
					if(!current_node)
					{
						current_node = new True_node();
						irr_beliefs = Trustee_node_r-> get_irr_beliefs();
						beliefs = Trustee_node_r -> get_belief_parameters();
						local_shifts = Trustee_node_r -> get_shifts();
						for(int l= 0; l < noT; ++l)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								current_node-> set_irr_parameters(irr, irr_beliefs[l][irr], l);
							}
							for(int b=0; b < nob; ++b)
							{
								current_node -> set_belief_parameters(b, beliefs[l][b], l);
							}
						}
						for(int level=0; level < noT+1; ++level)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								current_node -> set_shift(level, local_shifts[level][irr], irr);
							}
						}
						Trustee_node_r -> set_child(first_ret, current_node);
						for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
						{
							guilt_parameter[noT]=guilt_counter;
							investor_k( game+1
							, planning_horizon
							, guilt_parameter
							, -1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &current_node);

						}
						if((k2+2)%2== 0)
						{
							double sum = 0.0;
							double isum = 0.0;
							beliefs = current_node -> get_belief_parameters();
							irr_beliefs = current_node -> get_irr_beliefs();

							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[0][guilt_counter];
								sum += running_belief_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = irr_beliefs[0][guilt_counter];
								isum += running_irr_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
							}
							investor_expectation_calculation( 0, &current_node,  running_belief_probability , running_irr_probability);
						}
						else
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = 0.0;
							}
							running_irr_probability[0] = 1.0;
							investor_expectation_calculation( -1, &current_node,  running_belief_probability , running_irr_probability);
						}
					}
					irr_beliefs = Trustee_node_r-> get_irr_beliefs();
					beliefs = Trustee_node_r -> get_belief_parameters();
					local_shifts = Trustee_node_r -> get_shifts();
					for(int irr=0; irr < noi; ++irr)
					{
						current_node-> set_irr_parameters(irr, irr_beliefs[k2][irr], k2);
						if(k2>0)
						{
							current_node-> set_irr_parameters(irr, irr_beliefs[k2-1][irr], k2-1);
						}
					}
					for(int b=0; b < nob; ++b)
					{
						current_node -> set_belief_parameters(b, beliefs[k2][b], k2);
						if(k2>0)
						{
							current_node -> set_belief_parameters(b, beliefs[k2-1][b], k2-1);
						}
					}
					for(int irr=0; irr < noi; ++irr)
					{
						current_node -> set_shift(k2+1, local_shifts[k2+1][irr], irr);
						current_node -> set_shift(k2, local_shifts[k2][irr], irr);
					}
					double trustee_monetary_response = 1/6.0*static_cast<double>(rbf*5*money_invested*response[game]);
					if(!(money_invested==0))
					{
						if(k2>0 && money_invested>1)
						{
							belief_updates(&Trustee_node_r, &current_node, k2, trustee_monetary_response , response[game], false);
						}
						shift_updates(&Trustee_node_r, &current_node, k2, trustee_monetary_response);
					}
					local_shifts = current_node -> get_shifts();
					for(int I_guilt=0; I_guilt < nob; ++I_guilt)
					{
						if(!(current_node->get_confirm_exploration(I_guilt, k2)))
						{
							if(k2>0)
							{
								guilt_parameter[k2-1]=I_guilt;
							}
							guilt_parameter[noT]=I_guilt;
							investor_k( game+1
							, planning_horizon
							, guilt_parameter
							, k2-1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &current_node);
							current_node->confirm_exploration(I_guilt, k2);
						}

					}

					if(!current_node -> expectation_set(k2))
					{
						if(k2>0)
						{
							double sum = 0.0;
							double isum = 0.0;
							beliefs = current_node -> get_belief_parameters();
							irr_beliefs = current_node -> get_irr_beliefs();
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[k2][guilt_counter];
								sum += running_belief_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = irr_beliefs[k2][guilt_counter];
								isum += running_irr_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
							}
							investor_expectation_calculation( k2, &current_node,  running_belief_probability , running_irr_probability);
						}
						/*else
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/nob;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = 0.0;
							}
							running_irr_probability[0] = 1.0;
							investor_expectation_calculation( -1, &current_node,  running_belief_probability , running_irr_probability);
						}*/
					}

					loc_payoffs = current_node-> get_exp_payoffs(k2);
					inv_exp_payoffs = Irritation_Investor -> get_exp_payoffs(0);
					local_shifts = current_node -> get_shifts();

					for(int second_act = 0; second_act < noa; ++second_act)
					{
						action[game+1] = second_act;
						investor_monetary_action = 5.0*static_cast<double>(action[game+1]);
						Trustee_holder = current_node -> get_child(second_act);
						if(!Trustee_holder)
						{
							Trustee_holder = new True_node();
							beliefs = current_node -> get_belief_parameters();
							irr_beliefs = current_node -> get_irr_beliefs();
							local_shifts = current_node -> get_shifts();
							for(int l=0; l < noT; ++l)
							{
								for(int nirr=0; nirr < noi; ++nirr)
								{
									Trustee_holder-> set_irr_parameters(nirr, irr_beliefs[l][nirr], l);
								}
								for(int b=0; b < nob; ++b)
								{
									Trustee_holder-> set_belief_parameters(b, beliefs[l][b], l);
								}
							}
							for(int level=0; level < noT+1; ++level)
							{
								for(int nirr=0; nirr < noi; ++nirr)
								{
									Trustee_holder -> set_shift(level, local_shifts[level][nirr], nirr);
								}
							}
							current_node->set_child(action[game+1],Trustee_holder);						
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								guilt_parameter[noT]=guilt_counter;
								trustee_k(action[game+1]
								, game+1
								, planning_horizon
								, guilt_parameter
								, -1
								, action
								, response
								, ui_init_system
								, ut_init_system
								, expected_trustee_payoff
								, expected_outcome_utility
								, temperature
								, mempool
								, &Trustee_holder);


							}

							//level -1
							if((k2+2)%2==1)
							{
								sum =0.0;
								isum = 0.0;
								for(int o = 0; o < nob; ++o)
								{
									sum += beliefs[0][o];
								}
								for(int o = 0; o < noi; ++o)
								{
									isum += irr_beliefs[0][o];
								}
								for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = beliefs[0][guilt_counter]/sum;
								}
								for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
								{
									running_irr_probability[irr_counter] = irr_beliefs[0][irr_counter]/isum;
								}
								trustee_expectation_calculation( 0, &Trustee_holder,  running_belief_probability , running_irr_probability, action[game+1]);
							}
							else
							{
								for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
								}
								for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
								{
									running_irr_probability[irr_counter] = 0.0;
								}
								running_irr_probability[0]= 1.0;
								trustee_expectation_calculation( -1, &Trustee_holder, running_belief_probability , running_irr_probability, action[game+1]);
							}
						}

						beliefs = current_node -> get_belief_parameters();
						irr_beliefs = current_node -> get_irr_beliefs();
						local_shifts = current_node -> get_shifts();
						for(int nirr=0; nirr < noi; ++nirr)
						{
							Trustee_holder-> set_irr_parameters(nirr, irr_beliefs[k2][nirr], k2);
							if(k2>0)
							{
								Trustee_holder-> set_irr_parameters(nirr, irr_beliefs[k2-1][nirr], k2-1);
							}
						}
						for(int b=0; b < nob; ++b)
						{
							Trustee_holder-> set_belief_parameters(b, beliefs[k2][b], k2);
							if(k2>0)
							{
								Trustee_holder-> set_belief_parameters(b, beliefs[k2-1][b], k2-1);
							}
						}

						for(int nirr=0; nirr < noi; ++nirr)
						{
							Trustee_holder -> set_shift(k2+1, local_shifts[k2+1][nirr], nirr);
							Trustee_holder -> set_shift(k2, local_shifts[k2][nirr], nirr);
						}

						belief_updates(&current_node, &Trustee_holder, k2+1, investor_monetary_action , action[game+1], true);
						shift_updates(&current_node, &Trustee_holder, k2+1, investor_monetary_action);

						if(!(Trustee_holder->get_confirm_exploration(trustee_guilt, k2+1)))
						{
							trustee_k(action[game+1]
							, game+1
							, planning_horizon
							, guilt_parameter
							, k2
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &Trustee_holder);
							Trustee_holder->confirm_exploration(trustee_guilt, k2+1);
						}

						for(int irr=0; irr < noi; ++irr)
						{
							if(investor_irr_probability[irr]> 0.0001 )
							{
								for(int I_guilt=0; I_guilt < nob; ++I_guilt)
								{

									if(investor_belief_probability[I_guilt] > 0.001)
									{
										for(int second_return = 0; second_return < nor(second_act); ++second_return)
										{
											tru_prob = Trustee_holder-> get_exp_payoffs(k2+1);
											if(loc_payoffs[I_guilt][second_act]> 0.001 && (1.0-local_shifts[k2][irr]) > 0.001 && tru_prob[trustee_guilt][second_return] > 0.001)
											{
												t_util[first_ret] += investor_belief_probability[I_guilt]*(Trustee_holder-> get_payoff(second_return))*
												tru_prob[trustee_guilt][second_return]*investor_irr_probability[irr]*(1.0-local_shifts[k2][irr])*loc_payoffs[I_guilt][second_act];
											}
											if(inv_exp_payoffs[0][second_act]> 0.001 && local_shifts[k2][irr] > 0.001 && tru_prob[trustee_guilt][second_return] > 0.001)
											{
												t_util[first_ret] += investor_irr_probability[irr]*investor_belief_probability[I_guilt]*(Trustee_holder-> get_payoff(second_return))*
												tru_prob[trustee_guilt][second_return]*local_shifts[k2][irr]*inv_exp_payoffs[0][second_act];
											}
										}
									}
								}
							}
						}
					}

					Trustee_node_r -> set_payoff(first_ret, t_util[first_ret]);
				}

				//exp payoffs calc
				sum = 0.0;
				double max_util = -100.0;
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{
					max_util = (1.0/temperature*t_util[first_ret] > max_util ? 1.0/temperature*t_util[first_ret]: max_util);
				}
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{
					if(1.0/temperature*t_util[first_ret]-max_util > -20)
					{
						sum += exp(1.0/temperature*t_util[first_ret]);
					}

				}
				for(int first_ret = 0; first_ret < nor(money_invested); ++first_ret)
				{
					if(1.0/temperature*t_util[first_ret]-max_util > -20)
					{
						t_util[first_ret] = exp(1.0/temperature*t_util[first_ret])/sum;
					}
					else
					{
						t_util[first_ret] = 0.0;
					}
					Trustee_node_r -> set_exp_payoffs(first_ret, t_util[first_ret], trustee_guilt, k2+1);
				}
			//}
			Trustee_holder->confirm_exploration(trustee_guilt, k2+1);
		}

	}

}

void investor_k(  int const& present_time
					 , int const& planning_horizon
					 , std::array<int,noT+1> guilt_parameter
					 , int const& k1
					 , std::array<int, global_time_horizon> const& pastactions
					 , std::array<int, global_time_horizon> const& responses
					 , std::array<Matrix, nob> const& ui_init_system
					 , std::array<Matrix, nob> const& ut_init_system
					 , std::array<std::array<double, noa>, nob> const& expected_trustee_payoff
					 , std::array<std::array<std::array<double, noa>, nob>, nob> const& expected_outcome_utility
					 , double const& temperature
					, MEMORY_POOL<True_node>& mempool
					,  True_node** reference_node)
{
	std::vector<double> investor_val (noa*global_time_horizon,0.0);
	std::array<int, global_time_horizon> action;
	std::array<int, global_time_horizon> response;
	std::array<int, global_time_horizon> sorted_hist;
	int investor_guilt;
	std::array<int, 1> initial_count;
	std::array<int, 1> count;
	for(int l=0; l < global_time_horizon; ++l)
	{
		action[l]=pastactions[l];
	}

	for(int l=0; l < global_time_horizon; ++l)
	{
		response[l]=responses[l];
	}

	True_node* current_node;
	current_node = *reference_node;
	True_node* Trustee_node_r;
	Trustee_node_r = *reference_node;

	True_node* root = current_node;
	int game = present_time;
	if((k1 < 0) )
	{
		double sum = 0.0;
		double max_val = -100.0;
		int investor_guilt = guilt_parameter[noT];
		int horizon = 0;
		sum =0.0;
		max_val = -100.0;
		for(int action_loc =0; action_loc < noa; ++action_loc)
		{
			investor_val[action_loc+noa*horizon] = 0.0;

			for(int g =0; g < nob; ++g) //set payoff und set exp payoff
			{
				investor_val[action_loc+noa*horizon] += expected_outcome_utility[investor_guilt][g][action_loc]*1.0/3.0;
			}
			current_node -> set_payoff(action_loc, investor_val[action_loc+noa*horizon]);
			max_val = (max_val < investor_val[action_loc+noa*horizon] ? investor_val[action_loc+noa*horizon]:max_val);
		}

		for(int action_loc =0; action_loc < noa; ++action_loc) //change to local actions later
		{
			if(1.0/temperature*(investor_val[action_loc+noa*horizon]-max_val) > -20.0)
			{
				sum += exp(1.0/temperature*investor_val[action_loc+noa*horizon]);
			}

		}
		for(int action_loc =0; action_loc < noa; ++action_loc) //change to local actions later
		{
			if(1.0/temperature*(investor_val[action_loc+noa*horizon]-max_val) > -20.0)
			{
				investor_val[action_loc+noa*horizon] = exp(1.0/temperature*investor_val[action_loc+noa*horizon])/sum;
			}
			else
			{
				investor_val[action_loc+noa*horizon] = 0.0;
			}
			current_node-> set_exp_payoffs(action_loc,investor_val[action_loc+noa*horizon], investor_guilt, 0);
		}


		current_node-> confirm_exploration(investor_guilt, 0);
	}
	else
	{

		if((planning_horizon- game) == 1)
		{
			int investor_guilt = guilt_parameter[k1];
			std::array<std::array<double, nob>, noT> beliefs;
			beliefs = current_node -> get_belief_parameters();
			std::array<double,nob> trustee_belief_probability;
			std::array<double,nob> running_belief_probability;
			std::array<double,noi> running_irr_probability;
			std::array<std::array<double,noi>, noT> running_irr_beliefs;
			std::array<std::array<double, noi>, noT+1> local_shifts;
			if(k1>0)
			{
				double sum = 0.0;
				double isum = 0.0;
				beliefs = current_node -> get_belief_parameters();
				running_irr_beliefs = current_node -> get_irr_beliefs();
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = beliefs[k1-1][guilt_counter];
					sum += running_belief_probability[guilt_counter];
				}
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = running_irr_beliefs[k1-1][guilt_counter];
					isum += running_irr_probability[guilt_counter];
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
				}
				investor_expectation_calculation( k1-1, &current_node,  running_belief_probability , running_irr_probability);
			}
			/*else
			{
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = 0.0;
				}
				running_irr_probability[0] = 1.0;
				investor_expectation_calculation( -1, &current_node, running_belief_probability , running_irr_probability);
			}*/
			beliefs = current_node -> get_belief_parameters();
			running_irr_beliefs = current_node -> get_irr_beliefs();
			double sum = 0.0;
			for(int o = 0; o < nob; ++o)
			{
				sum += beliefs[k1][o];
			}
			for(int o = 0; o < nob; ++o)
			{
				trustee_belief_probability[o] = beliefs[k1][o]/sum;
			}
			double isum = 0.0;
			running_irr_beliefs = current_node -> get_irr_beliefs();
			for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
			{
				running_irr_probability[guilt_counter] = running_irr_beliefs[k1][guilt_counter];
				isum += running_irr_probability[guilt_counter];
			}
			for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
			{
				running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
			}
			double max_val = -100.0;
			int update=0;
			//for(int update=0; update < (planning_horizon-present_time); ++update)
			//{
				sum = 0.0;
				max_val = -100.0;
				double expectation = 0.0;
				if(k1>0)
				{
					expectation = current_node -> get_expectation(k1-1);
				}
				else
				{
					expectation = current_node -> get_expectation(0);
				}

				for(int i_action = 0; i_action < noa; ++i_action)
				{
					local_shifts = current_node -> get_shifts();
					investor_val[i_action+noa*update] = 0.0;
					double investor_monetary_action = 5.0*static_cast<double>(i_action);
					double disappointment =  expectation -  investor_monetary_action;
					for(int irr=0; irr < noi; ++irr)
					{
						local_shifts[k1][irr] = (disappointment > 0.0 ? 1.0/static_cast<double>(noi-1)*static_cast<double>(irr):(-1.0/static_cast<double>(noi-1)*static_cast<double>(irr))) + local_shifts[k1][irr];
						local_shifts[k1][irr] = (local_shifts[k1][irr] > 1.0 ? 1.0:local_shifts[k1][irr]);
						local_shifts[k1][irr] = (local_shifts[k1][irr] < 0.0 ? 0.0:local_shifts[k1][irr]);
					}

					for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
					{
						if(running_irr_probability[irr_counter]>0.001)
						{
							for(int g =0; g < nob; ++g) //set payoff und set exp payoff + shift of trustee
							{
								if(trustee_belief_probability[g]> 0.001 && (1.0- local_shifts[k1][irr_counter])>0.001)
								{
									investor_val[i_action+noa*update] += running_irr_probability[irr_counter]*(1.0- local_shifts[k1][irr_counter])*expected_outcome_utility[investor_guilt][g][i_action]*
									trustee_belief_probability[g];
								}
								if(local_shifts[k1][irr_counter] > 0.001 && trustee_belief_probability[g]> 0.001 )
								{
									investor_val[i_action+noa*update] += running_irr_probability[irr_counter]*local_shifts[k1][irr_counter]*expected_outcome_utility[investor_guilt][0][i_action]*trustee_belief_probability[g];//trustee_val[action_loc+noa*update]/sum;
								}
							}
						}
					}
					current_node -> set_payoff(i_action, investor_val[i_action+noa*update]);
					max_val = (max_val < investor_val[i_action+noa*update] ? investor_val[i_action+noa*update]:max_val);

				}
				for(int i_action = 0; i_action < noa; ++i_action)
				{
					if(1.0/temperature*(investor_val[i_action+noa*update]-max_val) > -20.0)
					{
						sum+= exp(1.0/temperature*investor_val[i_action+noa*update]);
					}

				}
				for(int i_action = 0; i_action < noa; ++i_action)
				{
					if(1.0/temperature*(investor_val[i_action+noa*update]-max_val) > -20.0)
					{
						investor_val[i_action+noa*update] = exp(1.0/temperature*investor_val[i_action+noa*update])/sum;
					}
					else
					{
						investor_val[i_action+noa*update] = 0.0;
					}
					current_node -> set_exp_payoffs(i_action, investor_val[i_action+noa*update], investor_guilt, k1+1);
				}
			//}

			current_node -> confirm_exploration(investor_guilt, k1+1);
		}
		if((planning_horizon- game) == 2)
		{
			int investor_guilt = guilt_parameter[k1];
			True_node* investor_holder;
			std::array<double, noa> prob;
			std::array<double, nob> prob_guilt;
			std::array<double, noi> prob_irr;
			std::array<std::array<double, nob>, noT> beliefs;
			std::array<std::array<double, nob>, noT> updated_beliefs;
			std::array<std::array<double, noa>, nob> tru_exp_payoffs;
			std::array<std::array<double, noa>, nob> repay_probability;
			std::array<std::array<double, noa>, nob> i_util;
			std::array<double,noi> investor_irr_probability;
			std::array<double,noi> running_irr_probability;
			std::array<double,nob> trustee_belief_probability;
			std::array<double,nob> running_belief_probability;
			std::array<std::array<double,noi>, noT> irr_beliefs;
			std::array<std::array<double, noi>, noT+1> local_shifts;
			beliefs = current_node -> get_belief_parameters();
			irr_beliefs = current_node -> get_irr_beliefs();
			local_shifts = current_node -> get_shifts();
			if(k1>0) //-1 is always explored & expectation already built
			{
				double sum = 0.0;
				double isum = 0.0;
				beliefs = current_node -> get_belief_parameters();
				irr_beliefs = current_node -> get_irr_beliefs();
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = beliefs[k1-1][guilt_counter];
					sum += running_belief_probability[guilt_counter];
				}
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = irr_beliefs[k1-1][guilt_counter];
					isum += running_irr_probability[guilt_counter];
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
				}
				investor_expectation_calculation( k1-1, &current_node,  running_belief_probability , running_irr_probability);
			}
			/*else
			{
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = 0.0;
				}
				running_irr_probability[0] = 1.0;
				investor_expectation_calculation( -1, &current_node, running_belief_probability , running_irr_probability);
			}*/			
			beliefs = current_node -> get_belief_parameters();
			irr_beliefs = current_node -> get_irr_beliefs();
			double isum = 0.0;
			for(int o = 0; o < noi; ++o)
			{
				isum += irr_beliefs[k1][o];
			}
			for(int o = 0; o < noi; ++o)
			{
				investor_irr_probability[o] = irr_beliefs[k1][o]/isum;
			}
			double sum =0.0;
			for(int o = 0; o < nob; ++o)
			{
				sum += beliefs[k1][o];
			}
			for(int o = 0; o < nob; ++o)
			{
				trustee_belief_probability[o] = beliefs[k1][o]/sum;
			}

			double b_sum =0.0;
			double i_sum = 0.0;
			sum = 0.0;
			double max_val = -100.0;

			//for(int update=1; update < (planning_horizon-present_time); ++update)
			//{
				for(int first_act = 0; first_act < noa; ++first_act)
				{
					Trustee_node_r = current_node -> get_child(first_act);
					action[game]=first_act; //followed by update
					if(!Trustee_node_r)
					{
						Trustee_node_r = new True_node();
						current_node -> set_child( first_act, Trustee_node_r);
						beliefs = current_node -> get_belief_parameters();
						irr_beliefs = current_node -> get_irr_beliefs();
						local_shifts = current_node -> get_shifts();
						for(int l=0; l < noT; ++l)
						{
							for(int b=0; b < nob; ++b)
							{
								Trustee_node_r-> set_belief_parameters(b, beliefs[l][b], l);
							}
							for(int irr=0; irr < noi; ++irr)
							{
								Trustee_node_r-> set_irr_parameters(irr, irr_beliefs[l][irr], l);
							}
						}
						for(int level=0; level < noT+1; ++level)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								Trustee_node_r -> set_shift(level, local_shifts[level][irr], irr);
							}
						}
						for(int guilt = 0; guilt < nob; ++guilt)
						{

							guilt_parameter[noT]=guilt;

							trustee_k(action[game]
							, game
							, planning_horizon
							, guilt_parameter
							, -1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &Trustee_node_r);
							//Trustee_node_r -> confirm_exploration(guilt, -1);

						}
						//level -1
						if((k1+2)%2==0)
						{
							sum =0.0;
							isum = 0.0;
							for(int o = 0; o < nob; ++o)
							{
								sum += beliefs[0][o];
							}
							for(int o = 0; o < noi; ++o)
							{
								isum += irr_beliefs[0][o];
							}
							for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[0][guilt_counter]/sum;
							}
							for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
							{
								running_irr_probability[irr_counter] = irr_beliefs[0][irr_counter]/isum;
							}
							trustee_expectation_calculation( 0, &Trustee_node_r,  running_belief_probability , running_irr_probability, action[game]);
						}
						else
						{
							for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
							}
							for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
							{
								running_irr_probability[irr_counter] = 0.0;
							}
							running_irr_probability[0]= 1.0;
							trustee_expectation_calculation( -1, &Trustee_node_r,  running_belief_probability , running_irr_probability, action[game]);
						}
					}
					beliefs = current_node -> get_belief_parameters();
					irr_beliefs = current_node -> get_irr_beliefs();
					local_shifts = current_node -> get_shifts();
					for(int b=0; b < nob; ++b)
					{
						if(k1>0)
						{
							Trustee_node_r-> set_belief_parameters(b, beliefs[k1-1][b], k1-1);
						}
						Trustee_node_r-> set_belief_parameters(b, beliefs[k1][b], k1);
					}
					for(int irr=0; irr < noi; ++irr)
					{
						if(k1 > 0)
						{
							Trustee_node_r-> set_irr_parameters(irr, irr_beliefs[k1-1][irr], k1-1);
						}
						Trustee_node_r-> set_irr_parameters(irr, irr_beliefs[k1][irr], k1);
					}
					for(int irr=0; irr < noi; ++irr)
					{
						Trustee_node_r -> set_shift(k1, local_shifts[k1][irr], irr);
						Trustee_node_r -> set_shift(k1+1, local_shifts[k1+1][irr], irr);
					}

					double investor_monetary_action = 5.0*static_cast<double>(first_act);
					if(k1>0)
					{
						belief_updates(&current_node, &Trustee_node_r, k1, investor_monetary_action, action[game], true);
					}
					shift_updates(&current_node, &Trustee_node_r, k1, investor_monetary_action);
					i_util[investor_guilt][first_act] = 0.0;

					for(int guilt = 0; guilt < nob; ++guilt)
					{
						if(k1 > 0)
						{
							guilt_parameter[k1-1]=guilt;
						}
						guilt_parameter[noT]=guilt;
						if(!(Trustee_node_r->get_confirm_exploration(guilt, k1)))
						{
							trustee_k(action[game]
							, game
							, planning_horizon
							, guilt_parameter
							, k1-1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &Trustee_node_r);
							Trustee_node_r -> confirm_exploration(guilt, k1);
						}
					}
					if(k1>0)
					{
						
						double sum = 0.0;
						double isum = 0.0;
						beliefs = Trustee_node_r -> get_belief_parameters();
						irr_beliefs = Trustee_node_r -> get_irr_beliefs();
						for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
						{
							running_belief_probability[guilt_counter] = beliefs[k1][guilt_counter];
							sum += running_belief_probability[guilt_counter];
						}
						for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
						{
							running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
						}
						for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
						{
							running_irr_probability[guilt_counter] = irr_beliefs[k1][guilt_counter];
							isum += running_irr_probability[guilt_counter];
						}
						for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
						{
							running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
						}
						trustee_expectation_calculation( k1, &Trustee_node_r,  running_belief_probability , running_irr_probability, first_act);
					}
					repay_probability = Trustee_node_r -> get_exp_payoffs(k1);
					tru_exp_payoffs = Trustee_node_r -> get_exp_payoffs(0);
					local_shifts = Trustee_node_r -> get_shifts();
					for(int first_ret = 0; first_ret < nor(first_act); ++first_ret)
					{
						response[game]=first_ret;
						for(int irr_counter=0; irr_counter < noi; ++irr_counter)
						{
							if(investor_irr_probability[irr_counter] > 0.001)
							{
								for(int guilt = 0; guilt < nob; ++guilt)
								{
									if(trustee_belief_probability[guilt] > 0.001 )
									{
										if((1.0-local_shifts[k1][irr_counter]) > 0.001 & repay_probability[guilt][first_ret] > 0.001)
										{
											i_util[investor_guilt][first_act] += (1.0-local_shifts[k1][irr_counter])*ui_init_system[investor_guilt](first_act, first_ret)*
											trustee_belief_probability[guilt]*repay_probability[guilt][first_ret]*investor_irr_probability[irr_counter];
										}
										if(local_shifts[k1][irr_counter] > 0.001 & tru_exp_payoffs[0][first_ret] > 0.001)
										{
											i_util[investor_guilt][first_act] += local_shifts[k1][irr_counter]*ui_init_system[investor_guilt](first_act, first_ret)*
											trustee_belief_probability[guilt]*tru_exp_payoffs[0][first_ret]*investor_irr_probability[irr_counter];
										}
									}
								}
							}
						}
						beliefs = Trustee_node_r -> get_belief_parameters();
						irr_beliefs = Trustee_node_r -> get_irr_beliefs();
						local_shifts = Trustee_node_r -> get_shifts();
						investor_holder = Trustee_node_r -> get_child(first_ret);
						if(!investor_holder)
						{
							investor_holder = new True_node();
							Trustee_node_r -> set_child( first_ret, investor_holder);

							for(int l=0; l < noT; ++l)
							{
								for(int b=0; b < nob; ++b)
								{
									investor_holder -> set_belief_parameters(b, beliefs[l][b], l);
								}
								for(int irr=0; irr < noi; ++irr)
								{
									investor_holder -> set_irr_parameters(irr, irr_beliefs[l][irr], l);
								}
							}
							for(int level=0; level < noT+1; ++level)
							{
								for(int irr=0; irr < noi; ++irr)
								{
									investor_holder -> set_shift(level, local_shifts[level][irr], irr);
								}
							}
							for(int guilt = 0; guilt < nob; ++guilt)
							{

								guilt_parameter[noT]=guilt;

								investor_k( game+1
								, planning_horizon
								, guilt_parameter
								, -1
								, action
								, response
								, ui_init_system
								, ut_init_system
								, expected_trustee_payoff
								, expected_outcome_utility
								, temperature
								, mempool
								, &investor_holder);
								//investor_holder -> confirm_exploration(guilt, -1);

							}

							//level -1
							if((k1+2)%2== 1)
							{
								double sum = 0.0;
								double isum = 0.0;
								beliefs = investor_holder -> get_belief_parameters();
								irr_beliefs = investor_holder -> get_irr_beliefs();

								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = beliefs[0][guilt_counter];
									sum += running_belief_probability[guilt_counter];
								}
								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
								}
								for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
								{
									running_irr_probability[guilt_counter] = irr_beliefs[0][guilt_counter];
									isum += running_irr_probability[guilt_counter];
								}
								for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
								{
									running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
								}
								investor_expectation_calculation( 0, &investor_holder,  running_belief_probability , running_irr_probability);
							}
							else
							{
								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
								}
								for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
								{
									running_irr_probability[guilt_counter] = 0.0;
								}
								running_irr_probability[0] = 1.0;
								investor_expectation_calculation( -1, &investor_holder, running_belief_probability , running_irr_probability);
							}
						}
						for(int b=0; b < nob; ++b)
						{
							if(k1>0)
							{
								investor_holder -> set_belief_parameters(b, beliefs[k1-1][b], k1-1);
							}
							investor_holder -> set_belief_parameters(b, beliefs[k1][b], k1);
						}
						for(int irr=0; irr < noi; ++irr)
						{
							if(k1>0)
							{
								investor_holder-> set_irr_parameters(irr, irr_beliefs[k1-1][irr], k1-1);
							}
								investor_holder-> set_irr_parameters(irr, irr_beliefs[k1][irr], k1);
						}
						for(int irr=0; irr < noi; ++irr)
						{
							investor_holder -> set_shift(k1, local_shifts[k1][irr], irr);
							investor_holder -> set_shift(k1+1, local_shifts[k1+1][irr], irr);
						}
						double trustee_monetary_action = 5.0/6.0*static_cast<double>(rbf*first_ret*first_act);
						if(!action[game]==0)
						{
							if(action[game]>1)
							{
								belief_updates(&Trustee_node_r, &investor_holder,  k1+1, trustee_monetary_action, response[game], false);
							}
							shift_updates(&Trustee_node_r, &investor_holder, k1+1, trustee_monetary_action);
						}
						//shift_updates(&Trustee_node_r, &investor_holder, k1, trustee_monetary_action);
						sum = 0.0;
						isum = 0.0;
						beliefs = investor_holder -> get_belief_parameters();
						irr_beliefs = investor_holder -> get_irr_beliefs();
						if(k1>0)
						{
							for(int guilt = 0; guilt < nob; ++guilt)
							{
								if(k1 > 1)
								{
									guilt_parameter[k1-2]=guilt;
								}
								guilt_parameter[noT]=guilt;
								if(!(investor_holder ->get_confirm_exploration(guilt, k1-1)))
								{
									investor_k( game+1
									, planning_horizon
									, guilt_parameter
									, k1-2
									, action
									, response
									, ui_init_system
									, ut_init_system
									, expected_trustee_payoff
									, expected_outcome_utility
									, temperature
									, mempool
									, &investor_holder);
									investor_holder -> confirm_exploration(guilt, k1-1);
								}
							}
						}
						if(k1 > 0)
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[k1-1][guilt_counter];
								sum += running_belief_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = irr_beliefs[k1-1][guilt_counter];
								isum += running_irr_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
							}
							investor_expectation_calculation( k1-1, &investor_holder,  running_belief_probability , running_irr_probability);
						}
						/*else
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = 0.0;
							}
							running_irr_probability[0] = 1.0;
							investor_expectation_calculation( k1-1, &investor_holder,  running_belief_probability , running_irr_probability);
						}*/

						beliefs = investor_holder -> get_belief_parameters();
						irr_beliefs = investor_holder -> get_irr_beliefs();
						b_sum =0.0;
						i_sum = 0.0;
						for(int believed_guilt =0; believed_guilt < nob; ++believed_guilt)
						{
							updated_beliefs[k1][believed_guilt] = beliefs[k1][believed_guilt];
							b_sum += updated_beliefs[k1][believed_guilt];
						}
						for(int believed_irr =0; believed_irr < noi; ++believed_irr)
						{
							i_sum += irr_beliefs[k1][believed_irr];
						}

						for(int i_action = 0; i_action < noa; ++i_action)
						{
							investor_val[i_action] = 0.0;
						}
						double expectation = 0.0;
						if(k1>0)
						{
							expectation = investor_holder->get_expectation(k1-1);
						}
						else
						{
							expectation = investor_holder->get_expectation(0);
						}
						for(int second_act = 0; second_act < noa; ++second_act)
						{
							local_shifts = investor_holder -> get_shifts();
							for(int believed_irr=0; believed_irr < noi; ++believed_irr)
							{
								prob_irr[believed_irr] = irr_beliefs[k1][believed_irr]/i_sum;
								double investor_monetary_action = 5.0*static_cast<double>(second_act);
								double disappointment =  expectation -  investor_monetary_action;
								local_shifts[k1][believed_irr] = (disappointment > 0.0 ? 1.0/static_cast<double>(noi-1)*static_cast<double>(believed_irr):(-1.0/static_cast<double>(noi-1)*static_cast<double>(believed_irr))) + local_shifts[k1][believed_irr];
								local_shifts[k1][believed_irr] = (local_shifts[k1][believed_irr] > 1.0 ? 1.0:local_shifts[k1][believed_irr]);
								local_shifts[k1][believed_irr] = (local_shifts[k1][believed_irr] < 0.0 ? 0.0:local_shifts[k1][believed_irr]);
								if(prob_irr[believed_irr]>0.001)
								{
									for(int believed_guilt =0; believed_guilt < nob; ++believed_guilt)
									{
										prob_guilt[believed_guilt] = updated_beliefs[k1][believed_guilt]/b_sum;
										if(prob_guilt[believed_guilt] > 0.001)
										{
											if((1.0-local_shifts[k1][believed_irr]) > 0.001 )
											{
												investor_val[second_act] += (1.0-local_shifts[k1][believed_irr])*(expected_outcome_utility[investor_guilt][believed_guilt][second_act]
												*prob_guilt[believed_guilt]*prob_irr[believed_irr]);
											}
											if(local_shifts[k1][believed_irr] > 0.001)
											{
												investor_val[second_act] += local_shifts[k1][believed_irr]*(expected_outcome_utility[investor_guilt][0][second_act]
												*prob_guilt[believed_guilt]*prob_irr[believed_irr]);
											}
										}
									}
								}
							}
						}
						sum =0.0;
						max_val  = -100.0;
						for(int second_act = 0; second_act < noa; ++second_act)
						{
							max_val = (max_val < 1.0/temperature*investor_val[second_act] ? 1.0/temperature*investor_val[second_act]:max_val);
						}
						for(int second_act = 0; second_act < noa; ++second_act)
						{
							if(1.0/temperature*investor_val[second_act] - max_val > -20.0)
							{
								sum += exp(1.0/temperature*investor_val[second_act]);
							}

						}
						for(int second_act = 0; second_act < noa; ++second_act)
						{
							if(1.0/temperature*investor_val[second_act] - max_val > -20.0)
							{
								prob[second_act] =  exp(1.0/temperature*investor_val[second_act])/sum;
							}
							else
							{
								prob[second_act] = 0.0;
							}
						}
						local_shifts = Trustee_node_r -> get_shifts();
						for(int believed_irr=0; believed_irr < noi; ++believed_irr)
						{
							for(int guilt = 0; guilt < nob; ++guilt)
							{
								if(trustee_belief_probability[guilt] > 0.001 & repay_probability[guilt][first_ret] > 0.001)
								{
									for(int second_act = 0; second_act < noa; ++second_act)
									{
										//if(prob[second_act] > 0.0001 )
										//{
											if((1.0-local_shifts[k1][believed_irr]) > 0.001 && repay_probability[guilt][first_ret]>0.001)
											{
												i_util[investor_guilt][first_act] += prob[second_act]*investor_val[second_act]*trustee_belief_probability[guilt]*investor_irr_probability[believed_irr]*
											(1.0-local_shifts[k1][believed_irr])*repay_probability[guilt][first_ret];
											}
											if(local_shifts[k1][believed_irr]>0.001 && tru_exp_payoffs[0][first_ret]>0.001)
											{
												i_util[investor_guilt][first_act] += prob[second_act]*investor_val[second_act]*trustee_belief_probability[guilt]*investor_irr_probability[believed_irr]*
												local_shifts[k1][believed_irr]*tru_exp_payoffs[0][first_ret];
											}
										//}
									}
								}
							}
						}

					}
					current_node -> set_payoff(first_act, i_util[investor_guilt][first_act]);

				}
				//}
				sum = 0.0;
				max_val  = -100.0;
				for(int first_act = 0; first_act < noa; ++first_act)
				{
					max_val = (max_val < 1.0/temperature*i_util[investor_guilt][first_act] ? 1.0/temperature*i_util[investor_guilt][first_act]:max_val);
				}
				for(int first_act = 0; first_act < noa; ++first_act)
				{
					if(1.0/temperature*i_util[investor_guilt][first_act]-max_val > -20.0)
					{
						sum += exp(1.0/temperature*i_util[investor_guilt][first_act]);
					}
				}
				for(int first_act = 0; first_act < noa; ++first_act)
				{
					if(1.0/temperature*i_util[investor_guilt][first_act]-max_val > -20.0)
					{
						i_util[investor_guilt][first_act] = exp(1.0/temperature*i_util[investor_guilt][first_act])/sum;
					}
					else
					{
						i_util[investor_guilt][first_act] = 0.0;
					}
					current_node -> set_exp_payoffs(first_act, i_util[investor_guilt][first_act], investor_guilt, k1+1);
				}
			//}

			current_node -> confirm_exploration(investor_guilt, k1+1);
		}

		if(planning_horizon-game > 2)
		{
			int investor_guilt = guilt_parameter[k1];
			True_node* investor_holder;
			std::array<double, noa> prob;
			std::array<double, nob> prob_guilt;
			std::array<double, noi> prob_irr;
			std::array<std::array<double, nob>, noT> beliefs;
			std::array<std::array<double, nob>, noT> updated_beliefs;
			std::array<std::array<double, noa>, nob> tru_exp_payoffs;
			std::array<std::array<double, noa>, nob> repay_probability;
			std::array<std::array<double, noa>, nob> hold_payoffs;
			std::array<std::array<double, noa>, nob> i_util;
			std::array<double,nob> investor_belief_probability;
			std::array<double,noi> investor_irr_probability;
			std::array<double,noi> running_irr_probability;
			std::array<double,nob> trustee_belief_probability;
			std::array<double,nob> running_belief_probability;
			std::array<std::array<double,noi>, noT> irr_beliefs;
			std::array<std::array<double, noi>, noT+1> local_shifts;
			beliefs = current_node -> get_belief_parameters();
			irr_beliefs = current_node -> get_irr_beliefs();
			local_shifts = current_node -> get_shifts();
			if(k1>0)
			{
				double sum = 0.0;
				double isum = 0.0;
				beliefs = current_node -> get_belief_parameters();
				irr_beliefs = current_node -> get_irr_beliefs();
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = beliefs[k1-1][guilt_counter];
					sum += running_belief_probability[guilt_counter];
				}
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = irr_beliefs[k1-1][guilt_counter];
					isum += running_irr_probability[guilt_counter];
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
				}
				investor_expectation_calculation( k1-1, &current_node, running_belief_probability , running_irr_probability);
			}
			/*else
			{
				for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
				{
					running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
				}
				for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
				{
					running_irr_probability[guilt_counter] = 0.0;
				}
				running_irr_probability[0] = 1.0;
				investor_expectation_calculation( -1, &current_node, running_belief_probability , running_irr_probability);
			}*/			
			beliefs = current_node -> get_belief_parameters();
			irr_beliefs = current_node -> get_irr_beliefs();
			double isum = 0.0;
			for(int o = 0; o < noi; ++o)
			{
				isum += irr_beliefs[k1][o];
			}
			for(int o = 0; o < noi; ++o)
			{
				investor_irr_probability[o] = irr_beliefs[k1][o]/isum;
			}
			double sum =0.0;
			for(int o = 0; o < nob; ++o)
			{
				sum += beliefs[k1][o];
			}
			for(int o = 0; o < nob; ++o)
			{
				trustee_belief_probability[o] = beliefs[k1][o]/sum;
			}

			double b_sum =0.0;
			double i_sum = 0.0;
			sum = 0.0;
			double max_val = -100.0;
			//for(int update = planning_horizon-game-1; update < (planning_horizon-present_time); ++update)
			//{

				for(int first_act = 0; first_act < noa; ++first_act)
				{
					action[game]=first_act; //followed by update
					double investor_monetary_action = 5.0*static_cast<double>(first_act);
					Trustee_node_r = current_node -> get_child(first_act);

					if(!Trustee_node_r)
					{
						Trustee_node_r = new True_node();
						current_node -> set_child( first_act, Trustee_node_r);
						beliefs = current_node -> get_belief_parameters();
						irr_beliefs = current_node -> get_irr_beliefs();
						local_shifts = current_node -> get_shifts();
						for(int l=0; l < noT; ++l)
						{
							for(int b=0; b < nob; ++b)
							{
								Trustee_node_r-> set_belief_parameters(b, beliefs[l][b], l);
							}
							for(int irr=0; irr < noi; ++irr)
							{
								Trustee_node_r-> set_irr_parameters(irr, irr_beliefs[l][irr], l);
							}
						}
						for(int level=0; level < noT+1; ++level)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								Trustee_node_r -> set_shift(level, local_shifts[level][irr], irr);
							}
						}
						for(int guilt = 0; guilt < nob; ++guilt)
						{

							guilt_parameter[noT]=guilt;

							trustee_k(action[game]
							, game
							, planning_horizon
							, guilt_parameter
							, -1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &Trustee_node_r);
							//Trustee_node_r -> confirm_exploration(guilt, -1);

						}

						//level -1
						if((k1+2)%2==0)
						{
							sum =0.0;
							isum = 0.0;
							for(int o = 0; o < nob; ++o)
							{
								sum += beliefs[0][o];
							}
							for(int o = 0; o < noi; ++o)
							{
								isum += irr_beliefs[0][o];
							}
							for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[0][guilt_counter]/sum;
							}
							for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
							{
								running_irr_probability[irr_counter] = irr_beliefs[0][irr_counter]/isum;
							}
							trustee_expectation_calculation( 0, &Trustee_node_r, running_belief_probability , running_irr_probability, action[game]);
						}
						else
						{
							for(int guilt_counter = 0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
							}
							for(int irr_counter = 0; irr_counter < noi; ++irr_counter)
							{
								running_irr_probability[irr_counter] = 0.0;
							}
							running_irr_probability[0]= 1.0;
							trustee_expectation_calculation( -1, &Trustee_node_r, running_belief_probability , running_irr_probability, action[game]);
						}
					}
					beliefs = current_node -> get_belief_parameters();
					irr_beliefs = current_node -> get_irr_beliefs();
					local_shifts = current_node -> get_shifts();
					for(int b=0; b < nob; ++b)
					{
						if(k1>0)
						{
							Trustee_node_r-> set_belief_parameters(b, beliefs[k1-1][b], k1-1);
						}
						Trustee_node_r-> set_belief_parameters(b, beliefs[k1][b], k1);
					}
					for(int irr=0; irr < noi; ++irr)
					{
						if(k1 > 0)
						{
							Trustee_node_r-> set_irr_parameters(irr, irr_beliefs[k1-1][irr], k1-1);
						}
						Trustee_node_r-> set_irr_parameters(irr, irr_beliefs[k1][irr], k1);
					}
					for(int irr=0; irr < noi; ++irr)
					{
						Trustee_node_r -> set_shift(k1, local_shifts[k1][irr], irr);
						Trustee_node_r -> set_shift(k1+1, local_shifts[k1+1][irr], irr);
					}

					if(k1>0 )
					{
						belief_updates(&current_node, &Trustee_node_r, k1, investor_monetary_action, action[game], true);
					}
					shift_updates(&current_node, &Trustee_node_r, k1, investor_monetary_action);

					i_util[investor_guilt][first_act] = 0.0;

					for(int guilt = 0; guilt < nob; ++guilt)
					{
						if(k1 > 0)
						{
							guilt_parameter[k1-1]=guilt;
						}
						guilt_parameter[noT]=guilt;
						if(!(Trustee_node_r->get_confirm_exploration(guilt, k1)))
						{
							trustee_k(action[game]
							, game
							, planning_horizon
							, guilt_parameter
							, k1-1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &Trustee_node_r);
							Trustee_node_r -> confirm_exploration(guilt, k1);
						}
					}

					double sum = 0.0;
					double isum = 0.0;
					beliefs = Trustee_node_r -> get_belief_parameters();
					irr_beliefs = Trustee_node_r -> get_irr_beliefs();
					for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
					{
						running_belief_probability[guilt_counter] = beliefs[k1][guilt_counter];
						sum += running_belief_probability[guilt_counter];
					}
					for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
					{
						running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
					}
					for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
					{
						running_irr_probability[guilt_counter] = irr_beliefs[k1][guilt_counter];
						isum += running_irr_probability[guilt_counter];
					}
					for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
					{
						running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
					}
					trustee_expectation_calculation( k1, &Trustee_node_r, running_belief_probability , running_irr_probability, first_act);

					repay_probability = Trustee_node_r -> get_exp_payoffs(k1);
					tru_exp_payoffs = Trustee_node_r -> get_exp_payoffs(0);
					local_shifts = Trustee_node_r -> get_shifts();
					for(int first_ret = 0; first_ret < nor(first_act); ++first_ret)
					{
						response[game]=first_ret;
						for(int irr_counter=0; irr_counter < noi; ++irr_counter)
						{
							if(investor_irr_probability[irr_counter] > 0.001)
							{
								for(int guilt = 0; guilt < nob; ++guilt)
								{
									if(trustee_belief_probability[guilt] > 0.001 )
									{
										if((1.0-local_shifts[k1][irr_counter]) > 0.001 & repay_probability[guilt][first_ret] > 0.001)
										{
											i_util[investor_guilt][first_act] += (1.0-local_shifts[k1][irr_counter])*ui_init_system[investor_guilt](first_act, first_ret)*
											trustee_belief_probability[guilt]*repay_probability[guilt][first_ret]*investor_irr_probability[irr_counter];
										}
										if(local_shifts[k1][irr_counter] > 0.001 & tru_exp_payoffs[0][first_ret] > 0.001)
										{
											i_util[investor_guilt][first_act] += local_shifts[k1][irr_counter]*ui_init_system[investor_guilt](first_act, first_ret)*
											trustee_belief_probability[guilt]*tru_exp_payoffs[0][first_ret]*investor_irr_probability[irr_counter];
										}
									}
								}
							}
						}
						beliefs = Trustee_node_r -> get_belief_parameters();
						irr_beliefs = Trustee_node_r -> get_irr_beliefs();
						local_shifts = Trustee_node_r -> get_shifts();
						investor_holder = Trustee_node_r -> get_child(first_ret);
						if(!investor_holder)
						{
							investor_holder = new True_node();
							Trustee_node_r -> set_child( first_ret, investor_holder);

							for(int l=0; l < noT; ++l)
							{
								for(int b=0; b < nob; ++b)
								{
									investor_holder -> set_belief_parameters(b, beliefs[l][b], l);
								}
								for(int irr=0; irr < noi; ++irr)
								{
									investor_holder -> set_irr_parameters(irr, irr_beliefs[l][irr], l);
								}
							}
							for(int level=0; level < noT+1; ++level)
							{
								for(int irr=0; irr < noi; ++irr)
								{
									investor_holder -> set_shift(level, local_shifts[level][irr], irr);
								}
							}
							for(int guilt = 0; guilt < nob; ++guilt)
							{

								guilt_parameter[noT]=guilt;

								investor_k(  game+1
								, planning_horizon
								, guilt_parameter
								, -1
								, action
								, response
								, ui_init_system
								, ut_init_system
								, expected_trustee_payoff
								, expected_outcome_utility
								, temperature
								, mempool
								, &investor_holder);
								//investor_holder -> confirm_exploration(guilt, -1);

							}

							if((k1+2)%2== 1)
							{
								double sum = 0.0;
								double isum = 0.0;
								beliefs = investor_holder -> get_belief_parameters();
								irr_beliefs = investor_holder -> get_irr_beliefs();

								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = beliefs[0][guilt_counter];
									sum += running_belief_probability[guilt_counter];
								}
								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
								}
								for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
								{
									running_irr_probability[guilt_counter] = irr_beliefs[0][guilt_counter];
									isum += running_irr_probability[guilt_counter];
								}
								for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
								{
									running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
								}
								investor_expectation_calculation( 0, &investor_holder, running_belief_probability , running_irr_probability);
							}
							else
							{
								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
								}
								for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
								{
									running_irr_probability[guilt_counter] = 0.0;
								}
								running_irr_probability[0] = 1.0;
								investor_expectation_calculation( -1, &investor_holder, running_belief_probability , running_irr_probability);
							}
						}
						for(int b=0; b < nob; ++b)
						{
							if(k1>0)
							{
								investor_holder -> set_belief_parameters(b, beliefs[k1-1][b], k1-1);
							}
							investor_holder -> set_belief_parameters(b, beliefs[k1][b], k1);
						}
						for(int irr=0; irr < noi; ++irr)
						{
							if(k1>0)
							{
								investor_holder-> set_irr_parameters(irr, irr_beliefs[k1-1][irr], k1-1);
							}
							investor_holder-> set_irr_parameters(irr, irr_beliefs[k1][irr], k1);
						}
						for(int irr=0; irr < noi; ++irr)
						{
							investor_holder -> set_shift(k1, local_shifts[k1][irr], irr);
							investor_holder -> set_shift(k1+1, local_shifts[k1+1][irr], irr);
						}
						double trustee_monetary_action = 5.0/6.0*static_cast<double>(rbf*first_ret*first_act);
						if(!action[game]==0)
						{	
							if(action[game]>1)
							{
								belief_updates(&Trustee_node_r, &investor_holder,  k1+1, trustee_monetary_action, response[game], false);
							}
							shift_updates(&Trustee_node_r, &investor_holder, k1+1, trustee_monetary_action);
						}
						//if(k1>0){
						//	shift_updates(&Trustee_node_r, &investor_holder, k1-1, trustee_monetary_action);
						//}
						
						sum = 0.0;
						isum = 0.0;
						beliefs = investor_holder -> get_belief_parameters();
						irr_beliefs = investor_holder -> get_irr_beliefs();
						if(k1>0)
						{
							for(int guilt = 0; guilt < nob; ++guilt)
							{
								if(k1 > 1)
								{
									guilt_parameter[k1-2]=guilt;
								}
								guilt_parameter[noT]=guilt;
								if(!(investor_holder ->get_confirm_exploration(guilt, k1-1)))
								{
									investor_k(  game+1
									, planning_horizon
									, guilt_parameter
									, k1-2
									, action
									, response
									, ui_init_system
									, ut_init_system
									, expected_trustee_payoff
									, expected_outcome_utility
									, temperature
									, mempool
									, &investor_holder);
									investor_holder -> confirm_exploration(guilt, k1-1);
								}
							}
						}

						if(k1 > 0)
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = beliefs[k1-1][guilt_counter];
								sum += running_belief_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = running_belief_probability[guilt_counter]/sum;
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = irr_beliefs[k1-1][guilt_counter];
								isum += running_irr_probability[guilt_counter];
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = running_irr_probability[guilt_counter]/isum;
							}
							investor_expectation_calculation( k1-1, &investor_holder, running_belief_probability , running_irr_probability);
						}
						/*else
						{
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
							}
							for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
							{
								running_irr_probability[guilt_counter] = 0.0;
							}
							running_irr_probability[0] = 1.0;
							investor_expectation_calculation( k1-1, &investor_holder, running_belief_probability , running_irr_probability);
						}*/
						
						if(!(investor_holder ->get_confirm_exploration(investor_guilt, k1+1)))
						{
							investor_k(  game+1
							, planning_horizon
							, guilt_parameter
							, k1
							, action
							, response
							, ui_init_system
							, ut_init_system
							, expected_trustee_payoff
							, expected_outcome_utility
							, temperature
							, mempool
							, &investor_holder);
							investor_holder -> confirm_exploration(investor_guilt, k1+1);
						}

						hold_payoffs = investor_holder-> get_exp_payoffs(k1+1);
						for(int i_action = 0; i_action < noa; ++i_action)
						{
							investor_val[i_action] = 0.0;
						}

						for(int second_act = 0; second_act < noa; ++second_act)
						{
							prob[second_act]=0.0;
							//if(hold_payoffs[investor_guilt][second_act] > 0.0001)
							//{
								investor_val[second_act] += (investor_holder -> get_payoff(second_act))*
								hold_payoffs[investor_guilt][second_act];
								prob[second_act] = hold_payoffs[investor_guilt][second_act];
							//}
						}
						sum =0.0;
						local_shifts = Trustee_node_r -> get_shifts();
						for(int believed_irr=0; believed_irr < noi; ++believed_irr)
						{
							for(int guilt = 0; guilt < nob; ++guilt)
							{
								if(trustee_belief_probability[guilt] > 0.001 & repay_probability[guilt][first_ret] > 0.001)
								{
									for(int second_act = 0; second_act < noa; ++second_act)
									{
										//if(prob[second_act] > 0.0001 )
										//{
											if((1.0-local_shifts[k1][believed_irr]) > 0.001 && repay_probability[guilt][first_ret]>0.001)
											{
												i_util[investor_guilt][first_act] += investor_val[second_act]*trustee_belief_probability[guilt]*investor_irr_probability[believed_irr]*
											(1.0-local_shifts[k1][believed_irr])*repay_probability[guilt][first_ret];
											}
											if(local_shifts[k1][believed_irr]>0.001 && tru_exp_payoffs[0][first_ret]>0.001)
											{
												i_util[investor_guilt][first_act] += investor_val[second_act]*trustee_belief_probability[guilt]*investor_irr_probability[believed_irr]*
												local_shifts[k1][believed_irr]*tru_exp_payoffs[0][first_ret];
											}
										//}
									}
								}
							}
						}

					}
					current_node -> set_payoff(first_act, i_util[investor_guilt][first_act]);

				}

				sum = 0.0;
				max_val  = -100.0;
				//red= -1000.0;
				for(int first_act = 0; first_act < noa; ++first_act)
				{
					max_val = (max_val < 1.0/temperature*i_util[investor_guilt][first_act] ? 1.0/temperature*i_util[investor_guilt][first_act]:max_val);
				}
				for(int first_act = 0; first_act < noa; ++first_act)
				{
					if(1.0/temperature*i_util[investor_guilt][first_act]-max_val > -20.0)
					{
						sum += exp(1.0/temperature*i_util[investor_guilt][first_act]);

					}
					/*else
					{
						red = (  red < 1.0/temperature*i_util[investor_guilt][first_act] ? 1.0/temperature*i_util[investor_guilt][first_act]:red );
					}*/
				}
				for(int first_act = 0; first_act < noa; ++first_act)
				{
					if(1.0/temperature*i_util[investor_guilt][first_act]-max_val > -20.0)
					{
						i_util[investor_guilt][first_act] = exp(1.0/temperature*i_util[investor_guilt][first_act])/(sum);
					}
					else
					{
						i_util[investor_guilt][first_act] = 0.0;
					}
					current_node -> set_exp_payoffs(first_act, i_util[investor_guilt][first_act], investor_guilt, k1+1);
				}
			//}
			investor_holder -> confirm_exploration(investor_guilt, k1+1);
		}

	}

}


int main(int argc, char* argv[])
{
	//---------------------------------------- storage variables for choice probabilities ("payoff") and expectations
	//std::array<double,noa>  investor_payoff;
	//std::array<double,noa>  trustee_payoff;
	static std::array<std::array<std::array<std::array<std::array<double,noa>,nob>, noi>, global_time_horizon>, noT>  hold_investor_payoffs;
	static std::array<std::array<std::array<std::array<double,noa>,nob>, noT> , global_time_horizon> store_investor_payoffs;
	static std::array<std::array<std::array<std::array<double,noa>,nob>, noT> , global_time_horizon> store_investor_shifted_payoffs;
	static std::array<std::array<std::array<std::array<std::array<double,noa>,nob>, noi>, global_time_horizon>, noT> WI_investor_payoffs;
	static std::array<std::array<std::array<std::array<std::array<double,noa>,nob>, noi>, global_time_horizon>, noT> hold_trustee_payoffs;
	static std::array<std::array<std::array<std::array<double,noa>,nob>, noT> , global_time_horizon> store_trustee_payoffs;
	static std::array<std::array<std::array<std::array<double,noa>,nob>, noT> , global_time_horizon> store_trustee_shifted_payoffs;
	static std::array<std::array<std::array<std::array<std::array<double,noa>,nob>, noi>, global_time_horizon>, noT> WT_trustee_payoffs;
	static std::array<std::array<double,noa>,nob> main_exp_payoffs;
	static std::array<std::array<double,noa>,nob> main_i_exp_payoffs;
	static std::array<std::array<double,noa>,nob> investor_payoffs;
	static std::array<std::array<double,noa>,nob> trustee_payoffs;
	static std::array<std::array<double,noa>,nob> shifted_payoffs;
	static std::array<std::array<double, noT>, global_time_horizon> store_investor_expectation;
	static std::array<std::array<double, noT> , global_time_horizon> store_trustee_expectation;
	static std::array<std::array<double, noT> , global_time_horizon> WI_investor_expectation;
	static std::array<std::array<double, noT> , global_time_horizon> WT_trustee_expectation;
	//--------------------------------------------------------------------------------------------------------------------------------
	double sum =0.0; //sum holder variable  investor
	double tsum = 0.0; //sum holder variable  trustee


	static int no_sub = 824;
	static int start_id;
	static int end_id;

//------------------------------------------------- readout block for prepared behaviour files ------------------------------------------------------------
	ifstream ifile ("/home/hulaa/work/Modelhold/Data/NSPNBaseExchange.bin", ios::in | ios::binary| ios::ate);	//
	ifstream::pos_type il;
	char * melblock;
    streampos size;
	size = ifile.tellg();
	ifile.seekg (0, ios::beg);
	melblock = new char [size];

	ifile.read (melblock, size);
	ifile.close();
	il =0;

	static std::array<std::array<std::array<int,global_time_horizon>,2>,1000> Subject_Games;	//set-up of subject interaction block

	for(int sub_id=0; sub_id < no_sub; ++sub_id)
	{
		for(int turn=0; turn <global_time_horizon; ++turn)
		{
			Subject_Games[sub_id][0][turn]= *(int*)&melblock[il];
			il += sizeof *(int*)&melblock[il];
			Subject_Games[sub_id][1][turn]=*(int*)&melblock[il];
			il += sizeof *(int*)&melblock[il];
		}
	}

	delete[] melblock;
//-------------------------------------------------------------------------------------------------------------------
	double trustee_monetary_response;	//holder for current trustee response in terms of monetary value
	double investor_monetary_action;  //holder for current investor response in terms of monetary value
	// ------------------------------------- belief holder variables: Guilt beliefs and Irritability beliefs for both actors ----------------------------
	static std::array<std::array<double, nob>, noT> trustee_belief_parameters;
	static std::array<std::array<double, nob>, noT> investor_belief_parameters;
	static std::array<std::array<double, noi>, noT> trustee_irr_beliefs;
	static std::array<std::array<double, noi>, noT> investor_irr_beliefs;

	//----------------------------------- model and output holder variables --------------------------------------
	static std::array<std::array<int,2>,8100> Subject_Guilt; //store subject guilt values (8100 as arbitrary limit)
	static std::array<std::array<int,2>,8100> Subject_ToM;		//store subject guilt values (8100 as arbitrary limit)
	static double investor_temperature;	//softmax temperature for investor
	static double trustee_temperature;  //softmax temperature for trustee
	int investor_planning; //planning limit investor
	int trustee_planning; // planning limit trustee
	int d1; //planning steps variable investor
	int d2; //planning steps variable trustee
	static int WI_ToM; //storage for winning model ToM - investor
	static int WI_G; //storage for winning model guilt - investor
	static int WI_P; //storage for winning model planning - investor
	static int WT_ToM; //storage for winning model ToM - trustee
	static int WT_G; //storage for winning model guilt - trustee
	static int WT_P; //storage for winning model planning - trustee
	static int WI_risk; //storage for winning model risk aversion - investor
	static int WT_risk; //storage for winning model risk aversion - trustee
	static int WI_irritability; //storage for winning model irritability - investor
	static int WT_irritability; //storage for winning model irritability - trustee
	static double WI_temp; //storage for winning model Temperature - investor
	static double WT_temp; //storage for winning model Temperature - trustee
	static int WI_shift; //storage for winning model irritation belief - investor
	static int WT_shift; //storage for winning model irritation belief - trustee
	static double investor_irritation;
	static double trustee_irritation;

	static double investor_shifted;
	static double trustee_shifted;

	static double Tlike_max;
	static double Ilike_max;
	static std::array<std::array<std::array<double, nob>, noT+1>, noi>  like;
	static std::array<std::array<std::array<double, nob>, noT+1>, noi>  Tlike;

	vector<double> holder(global_time_horizon*noa,0.0);
	static Matrix investor_ui_init;
	static Matrix_vector investor_utpi_init;
	static Matrix_vector trustee_utpi_init;
	static std::array< Matrix , nob> investor_trustee_probabilities;
	static std::array< Matrix , nob> trustee_trustee_probabilities;

	std::array<int, noT+1> guilt_parameter;
	static std::array<Matrix, nob> investor_ui_init_system;
	static std::array<Matrix, nob> investor_ut_init_system;

	static std::array<Matrix, nob> trustee_ui_init_system;
	static std::array<Matrix, nob> trustee_ut_init_system;

	static std::array<std::array<double, noa>, nob> investor_expected_trustee_payoff;
	static std::array<std::array<std::array<double, noa>, nob>, nob> investor_expected_outcome_utility;
	static std::array<std::array<double, noa>, nob> trustee_expected_trustee_payoff;
	static std::array<std::array<std::array<double, noa>, nob>, nob> trustee_expected_outcome_utility;

	static std::array<int,4> Planning;
	Planning[0]=1;
	Planning[1]=2;
	Planning[2]=3;
	Planning[3]=4;
	double inv_risk_aversion;
	double tru_risk_aversion;

	static std::array<std::array<std::array<double, nob>, noT>,global_time_horizon> WT_trustee_belief_parameters;
	static std::array<std::array<std::array<double, nob>, noT>,global_time_horizon> WI_investor_belief_parameters;
	static std::array<std::array<std::array<double, noi>, noT>,global_time_horizon> WT_irr_beliefs;
	static std::array<std::array<std::array<double, noi>, noT>,global_time_horizon> WI_irr_beliefs;
	static std::array<std::array<std::array<double, noi>, noT>,global_time_horizon> store_investor_irr_beliefs;
	static std::array<std::array<std::array<double, noi>, noT>,global_time_horizon> store_trustee_irr_beliefs;


	static std::array<double, 8> risk_grid;
	risk_grid[0] = 0.4;
	risk_grid[1] = 0.6;
	risk_grid[2] = 0.8;
	risk_grid[3] = 1.0;
	risk_grid[4] = 1.2;
	risk_grid[5] = 1.4;
	risk_grid[6] = 1.6;
	risk_grid[7] = 1.8;

	static std::array<std::array<double, noi>,5> irr_belief_presets;
	irr_belief_presets[0][0] = 400.1;
	irr_belief_presets[0][1] = 0.1;
	irr_belief_presets[0][2] = 0.1;
	irr_belief_presets[0][3] = 0.1;
	irr_belief_presets[0][4] = 0.1;
	irr_belief_presets[1][0] = 4.0;
	irr_belief_presets[1][1] = 0.5;
	irr_belief_presets[1][2] = 0.5;
	irr_belief_presets[1][3] = 0.5;
	irr_belief_presets[1][4] = 0.5;
	irr_belief_presets[2][0] = 0.4;
	irr_belief_presets[2][1] = 0.1;
	irr_belief_presets[2][2] = 0.1;
	irr_belief_presets[2][3] = 0.1;
	irr_belief_presets[2][4] = 0.1;
	irr_belief_presets[3][0] = 2.0;
	irr_belief_presets[3][1] = 1.0;
	irr_belief_presets[3][2] = 1.0;
	irr_belief_presets[3][3] = 1.0;
	irr_belief_presets[3][4] = 1.0;
	irr_belief_presets[4][0] = 0.1;
	irr_belief_presets[4][1] = 0.1;
	irr_belief_presets[4][2] = 0.1;
	irr_belief_presets[4][3] = 0.1;
	irr_belief_presets[4][4] = 400.1;
	static std::array<double, 4> temperature_grid;
	temperature_grid[0]= 4.0;
	temperature_grid[1]= 3.0;
	temperature_grid[2]= 2.0;
	temperature_grid[3]= 1.0;

	static std::array<std::array<double, noi>, noT+1> investor_shifts;
	static std::array<std::array<double, noi>, noT+1> trustee_shifts;
	static std::array<std::array<std::array<double, noi>, noT+1>, global_time_horizon> store_investor_shifts;
	static std::array<std::array<std::array<double, noi>, noT+1>, global_time_horizon> store_trustee_shifts;
	static std::array<std::array<std::array<double, noi>, noT+1>, global_time_horizon> WI_investor_shifts;
	static std::array<std::array<std::array<double, noi>, noT+1>, global_time_horizon> WT_trustee_shifts;
	static std::array<double, nob>  main_running_belief_probability;
	static std::array<double, noi>  main_irr_probability;

	True_node* Trust;
	True_node* temp_node;

	static std::array<int, global_time_horizon> hold_actions; //output variable. store current investor choices
	static std::array<int, global_time_horizon> hold_responses;	//output variable. store current trustee choices
	static std::array<std::array<std::array<double, nob>, noT>,global_time_horizon> store_trustee_belief_parameters;	//output variable to keep prior belief values on guilt of investor
	static std::array<std::array<std::array<double, nob>, noT>,global_time_horizon> store_investor_belief_parameters; //output variable to keep prior belief values on guilt of trustee

	/*int investor_irr = 0;	//generative model mode holders for irritation settings
	int trustee_irr = 0;*/
	MEMORY_POOL<True_node> mexpool;
	ofstream ofs("/home/hulaa/work/Modelhold/DeGNSPNMod3.bin", ofstream::out| ofstream::binary);			//output stream
	start_id= 309;
	end_id = 412;
	True_node* investor_node;	//pointer to current investor node
	True_node* trustee_node;	//pointer to current trustee node
	int countr = 0;
	static int investor_choice_holder = 0;
	static int trustee_choice_holder = 0;
	//adapt belief updates from above
	inv_risk_aversion = 1.0;
	investor_temperature =  2.0;
	investor_utpi_init = initialize_utpi_init(1.0);

	for(int system=0; system < nob; ++system)
	{
		investor_ui_init_system[system] = initialize_ui_init(system, inv_risk_aversion);
	}
	for(int system=0; system < nob; ++system)
	{
		investor_ut_init_system[system] = initialize_trustee_utility(system, 1.0);
		investor_trustee_probabilities[system]=initialize_trustee_probabilities(investor_ut_init_system[system], investor_temperature);
	}
	investor_expected_trustee_payoff  = initialize_expected_trustee_payoff(investor_trustee_probabilities, investor_ut_init_system);
	investor_expected_outcome_utility = initialize_expected_outcome_utility(investor_trustee_probabilities, investor_ui_init_system);

	for(int sub_id= start_id; sub_id < end_id; ++sub_id)
	{
		Subject_ToM[sub_id][0]=4;
		Subject_ToM[sub_id][1]=3;

		Tlike_max = 1000;
		Ilike_max = 1000;
		cout << sub_id <<endl; //to track progress

		for(int risk_count= 0; risk_count < 8; ++risk_count)
		{

			inv_risk_aversion = risk_grid[risk_count];
			tru_risk_aversion = risk_grid[risk_count];

			for(int temp_count=0; temp_count < 4; ++temp_count)
			{
				investor_temperature =  temperature_grid[temp_count];
				trustee_temperature = temperature_grid[temp_count];
				if(inv_risk_aversion < 1.2)
				{
					inv_risk_aversion = 1.0;
					investor_utpi_init = initialize_utpi_init(1.0);
					//if(investor_temperature > 2.0 ){
					//	investor_temperature=2.0;
					//}
					for(int system=0; system < nob; ++system)
					{
						investor_ui_init_system[system] = initialize_ui_init(system, inv_risk_aversion);
					}

					for(int system=0; system < nob; ++system)
					{
						investor_ut_init_system[system] = initialize_trustee_utility(system, 1.0);
						investor_trustee_probabilities[system]=initialize_trustee_probabilities(investor_ut_init_system[system], investor_temperature);
					}
					investor_expected_trustee_payoff  = initialize_expected_trustee_payoff(investor_trustee_probabilities, investor_ut_init_system);
					investor_expected_outcome_utility = initialize_expected_outcome_utility(investor_trustee_probabilities, investor_ui_init_system);
					for(int guilt_counter = 0; guilt_counter < nob ; ++guilt_counter)
					{
						guilt_parameter[noT]=guilt_counter;
						investor_k(  0
						, 1
						, guilt_parameter
						, -1
						, Subject_Games[sub_id][0]
						, Subject_Games[sub_id][1]
						, investor_ui_init_system
						, investor_ut_init_system
						, investor_expected_trustee_payoff
						, investor_expected_outcome_utility
						, investor_temperature
						, mexpool
						, &Irritation_Investor);
					}
					inv_risk_aversion = risk_grid[risk_count];
					investor_temperature =  temperature_grid[temp_count];
				}
				if(inv_risk_aversion > 1.0)
				{
					investor_utpi_init = initialize_utpi_init(1.0);
					//if(investor_temperature > 2.0 ){
					//	investor_temperature=2.0;
					//}
					for(int system=0; system < nob; ++system)
					{
						investor_ui_init_system[system] = initialize_ui_init(system, inv_risk_aversion);
					}

					for(int system=0; system < nob; ++system)
					{
						investor_ut_init_system[system] = initialize_trustee_utility(system, 1.0);
						investor_trustee_probabilities[system]=initialize_trustee_probabilities(investor_ut_init_system[system], investor_temperature);
					}
					investor_expected_trustee_payoff  = initialize_expected_trustee_payoff(investor_trustee_probabilities, investor_ut_init_system);
					investor_expected_outcome_utility = initialize_expected_outcome_utility(investor_trustee_probabilities, investor_ui_init_system);
					for(int guilt_counter = 0; guilt_counter < nob ; ++guilt_counter)
					{
						guilt_parameter[noT]=guilt_counter;
						investor_k(  0
						, 1
						, guilt_parameter
						, -1
						, Subject_Games[sub_id][0]
						, Subject_Games[sub_id][1]
						, investor_ui_init_system
						, investor_ut_init_system
						, investor_expected_trustee_payoff
						, investor_expected_outcome_utility
						, investor_temperature
						, mexpool
						, &Irritation_Investor);
					}
					investor_temperature =  temperature_grid[temp_count];					
				}				
				investor_utpi_init = initialize_utpi_init(1.0);


				for(int system=0; system < nob; ++system)
				{
					investor_ui_init_system[system] = initialize_ui_init(system, inv_risk_aversion);
				}

				for(int system=0; system < nob; ++system)
				{
					investor_ut_init_system[system] = initialize_trustee_utility(system, 1.0);
					investor_trustee_probabilities[system]=initialize_trustee_probabilities(investor_ut_init_system[system], investor_temperature);
				}
				investor_expected_trustee_payoff  = initialize_expected_trustee_payoff(investor_trustee_probabilities, investor_ut_init_system);
				investor_expected_outcome_utility = initialize_expected_outcome_utility(investor_trustee_probabilities, investor_ui_init_system);

				trustee_utpi_init = initialize_utpi_init(1.0);
				for(int system=0; system < nob; ++system)
				{
					trustee_ui_init_system[system] = initialize_ui_init(system, tru_risk_aversion);
				}
				for(int system=0; system < nob; ++system)
				{
					trustee_ut_init_system[system] = initialize_trustee_utility(system, 1.0);
					trustee_trustee_probabilities[system] =initialize_trustee_probabilities(trustee_ut_init_system[system], trustee_temperature);
				}
				trustee_expected_trustee_payoff  = initialize_expected_trustee_payoff(trustee_trustee_probabilities, trustee_ut_init_system);
				trustee_expected_outcome_utility = initialize_expected_outcome_utility(trustee_trustee_probabilities, trustee_ui_init_system);


				for(int plan_count=0; plan_count < 4; ++plan_count)
				{
					d1 =Planning[plan_count]; //get current planning var
					d2 =Planning[plan_count];
					for(int irr_belief=0; irr_belief < 1; ++irr_belief)
					{

						for(int level=0; level < noT; ++level)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								investor_irr_beliefs[level][irr] = 0.1; //set default irritation beliefs, in case
							}
							for(int belief=0; belief < nob; ++belief)
							{
								investor_belief_parameters[level][belief] = 1.0;// set initial guilt beliefs

								for(int time_step=0; time_step < global_time_horizon; ++time_step)//initialize storage array
								{
									store_investor_belief_parameters[time_step][level][belief]=investor_belief_parameters[level][belief];
									store_investor_expectation[time_step][level] = 0.0;
								}
							}
						}

						for(int level=0; level < noT; ++level)
						{
							for(int irr=0; irr < noi; ++irr)
							{
								trustee_irr_beliefs[level][irr] = 0.1; //set default irritation beliefs, in case
							}
							for(int belief=0; belief < nob; ++belief)
							{
								trustee_belief_parameters[level][belief] = 1.0;	// set initial guilt beliefs
								for(int time_step=0; time_step < global_time_horizon; ++time_step)	//initialize storage array
								{
									store_trustee_belief_parameters[time_step][level][belief]=trustee_belief_parameters[level][belief];
									store_trustee_expectation[time_step][level]=0.0;
								}
							}
						}
						for(int irr =0; irr < noi; ++irr)
						{
							for(int shift_reset =0; shift_reset < noT+1; ++shift_reset)
							{
								investor_shifts[shift_reset][irr]= 0.0; //assume no one starts the game irritated
								trustee_shifts[shift_reset][irr] = 0.0;
							}
						}

						for(int level = 0; level < noT; ++level)
						{
							for(int irr =0; irr < noi; ++irr)
							{
								investor_irr_beliefs[level][irr] = irr_belief_presets[irr_belief][irr]; //use a preset prior according to irritation belief settings
								trustee_irr_beliefs[level][irr] = irr_belief_presets[irr_belief][irr];
							}
						}

						for(int irr =0; irr < noi; ++irr)
						{
							for(int bel=0; bel < nob; ++bel)
							{
								for(int level = 0; level < noT; ++level)
								{
									like[irr][level][bel] = 0.0;	//initialize Negative-Log-Likelihood holders - investor
									Tlike[irr][level][bel] = 0.0;  //initialize Negative-Log-Likelihood holders - trustee
								}
							}
						}

						for(int time_step=0; time_step < global_time_horizon; ++time_step)	//global_time_horizon
						{
							MEMORY_POOL<True_node> mempool; //start local mempool
							investor_planning = mini(time_step+d1+1, global_time_horizon);	//determine local planning horizon
							trustee_planning= mini(time_step+d2+1, global_time_horizon);
							investor_node = new True_node(); //stays at root. Symbolizes investor at beginning of round
							trustee_node = new True_node(); //stays at root.  Symbolizes trustee at beginning of round

							for(int level=0; level < noT; ++level)  //initialize priors on guilt beliefs on the new nodes
							{
								for(int belief=0; belief < nob; ++belief)
								{
									investor_node-> set_belief_parameters(belief, investor_belief_parameters[level][belief], level);
									trustee_node-> set_belief_parameters(belief, trustee_belief_parameters[level][belief], level);
								}
							}

							for(int level=0; level < noT; ++level)  //initialize priors on irritation beliefs on the new nodes
							{
								for(int irr=0; irr < noi; ++irr)
								{
									investor_node-> set_irr_parameters(irr, investor_irr_beliefs[level][irr], level);
									store_investor_irr_beliefs[time_step][level][irr] = investor_irr_beliefs[level][irr];
									trustee_node-> set_irr_parameters(irr, trustee_irr_beliefs[level][irr], level);
									store_trustee_irr_beliefs[time_step][level][irr] = trustee_irr_beliefs[level][irr];
								}
							}

							//set irritation shifts on the new nodes
							for(int level=0; level < noT+1; ++level)
							{
								for(int irr=0; irr < noi; ++irr)
								{
									investor_node -> set_shift(level, investor_shifts[level][irr], irr);
									store_investor_shifts[time_step][level][irr] = investor_shifts[level][irr];
									trustee_node -> set_shift(level, trustee_shifts[level][irr], irr);
									store_trustee_shifts[time_step][level][irr] = trustee_shifts[level][irr];
								}
							}
							//------------------------ Level -1 set-up block --------------------------------------
							for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
							{
								//trustee model of level -1 investor choice probabilities for all investor guilt types
								guilt_parameter[noT]=guilt_counter;
								investor_k( time_step
								, trustee_planning
								, guilt_parameter
								, -1
								, Subject_Games[sub_id][0]
								, Subject_Games[sub_id][1]
								, trustee_ui_init_system
								, trustee_ut_init_system
								, trustee_expected_trustee_payoff
								, trustee_expected_outcome_utility
								, trustee_temperature
								, mempool
								, &trustee_node);
								//investor model of level -1 investor choice probabilities for all investor guilt types
								guilt_parameter[noT]=guilt_counter;
								investor_k(  time_step
								, investor_planning
								, guilt_parameter
								, -1
								, Subject_Games[sub_id][0]
								, Subject_Games[sub_id][1]
								, investor_ui_init_system
								, investor_ut_init_system
								, investor_expected_trustee_payoff
								, investor_expected_outcome_utility
								, investor_temperature
								, mempool
								, &investor_node);
							}

							if(!trustee_node-> expectation_set(0))
							{			//set up expectation
								if((Subject_ToM[sub_id][1]+3)%2== 1)
								{ 		//if -1 investor is part of the ToM hierarchy of the trustee, then beliefs of level 0 trustee on investor need to be accounted for
									double sum = 0.0;
									double isum = 0.0;
									//calculate probabilities on investor guilt and irritability from the prior beliefs
									trustee_belief_parameters = trustee_node -> get_belief_parameters();
									trustee_irr_beliefs = trustee_node -> get_irr_beliefs();
									for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
									{
										main_running_belief_probability[guilt_counter] = trustee_belief_parameters[0][guilt_counter];
										sum += main_running_belief_probability[guilt_counter];
									}
									for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
									{
										main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
									}
									for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
									{
										main_irr_probability[guilt_counter] = trustee_irr_beliefs[0][guilt_counter];
										isum += main_irr_probability[guilt_counter];
									}
									for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
									{
										main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
									}
									//calculate the expected value of the investor action from the point of view of a level 0 trustee
									investor_expectation_calculation( 0, &trustee_node, main_running_belief_probability , main_irr_probability);

								}
								else
								{
									//if level -1 investor is not part of the ToM hierarchy (for instancen for a level 1 trustee)
									//then the resulting investor level-2 has a simplified calculation
									for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
									{	//all guilt states are equally likely
										main_running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
									}
									for(int irr_counter=0; irr_counter < noi; ++irr_counter)
									{	//partner is not modelled as irritable
										main_irr_probability[irr_counter] = 0.0;
									}
									main_irr_probability[0]= 1.0;
									//calculate the expected value of the investor action from the point of view of a level -1 trustee
									investor_expectation_calculation( -1, &trustee_node,  main_running_belief_probability , main_irr_probability);
								}
							}
							if(!investor_node-> expectation_set(0))
							{
								//investor model of trustee expectation of investor action
								if((Subject_ToM[sub_id][0]+3)%2== 0)
								{
									//if investor of level -1 is part of the ToM hierarchy (for instance level 1 investor -> less common hierarchy)
									double sum = 0.0;
									double isum = 0.0;
									investor_belief_parameters = investor_node -> get_belief_parameters();
									investor_irr_beliefs = investor_node -> get_irr_beliefs();
									for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
									{
										main_running_belief_probability[guilt_counter] = investor_belief_parameters[0][guilt_counter];
										sum += main_running_belief_probability[guilt_counter];
									}
									for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
									{
										main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
									}
									for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
									{
										main_irr_probability[guilt_counter] = investor_irr_beliefs[0][guilt_counter];
										isum += main_irr_probability[guilt_counter];
									}
									for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
									{
										main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
									}
									investor_expectation_calculation( 0, &investor_node, main_running_belief_probability , main_irr_probability);
								}
								else
								{
									//if investor of level -1 is not part of the ToM hierarchy (for instance level 0 or level 2 investor -> common hierarchy)
									for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
									{
										main_running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
									}
									for(int irr_counter=0; irr_counter < noi; ++irr_counter)
									{
										main_irr_probability[irr_counter] = 0.0;
									}
									main_irr_probability[0]= 1.0;
									investor_expectation_calculation( -1, &investor_node,  main_running_belief_probability , main_irr_probability);
								}
							}
							//-------------------- sub-block level -1 trustee
							for(int first_action = 0; first_action < noa; ++first_action)	//explore all initial investor actions
							{
								hold_actions[time_step]=first_action;
								double investor_monetary_action = 5.0*static_cast<double>(first_action);
								//---------------------------- investor ----------------------------------------------
								Trust = new True_node();	//set child node for this action history
								investor_belief_parameters = investor_node -> get_belief_parameters();	//initialize all guilt beliefs in new node
								for(int level=0; level < noT; ++level)
								{
									for(int belief=0; belief < nob; ++belief)
									{
										Trust-> set_belief_parameters(belief, investor_belief_parameters[level][belief], level);
									}
								}
								investor_irr_beliefs = investor_node -> get_irr_beliefs(); //initialize all irritation beliefs in new node
								for(int level=0; level < noT; ++level)
								{
									for(int irr=0; irr < noi; ++irr)
									{
										Trust-> set_irr_parameters(irr, investor_irr_beliefs[level][irr], level);
									}
								}
								investor_shifts = investor_node -> get_shifts(); //initialize all irritation shifts in new node
								for(int level=0; level < noT+1; ++level)
								{
									for(int irr=0; irr < noi; ++irr)
									{
										Trust -> set_shift(level, investor_shifts[level][irr], irr);
									}
								}
								if((Subject_ToM[sub_id][0]+3)%2== 1)
								{	//if level -1 not part of hierarchy -> no belief update (but still shift update)
									shift_updates(&investor_node, &Trust, 0, investor_monetary_action);
									investor_shifts = Trust -> get_shifts();
									for(int irr=0; irr < noi; ++irr)
									{
										store_investor_shifts[time_step][0][irr] = investor_shifts[0][irr];
									}
								}
								if((Subject_ToM[sub_id][0]+3)%2== 0)
								{	//if level -1 not part of hierarchy -> all updates (i.e. a level 0 is doing inference on them)
									//belief_updates(&investor_node, &Trust, 1, investor_monetary_action , first_action, true);
									belief_updates(&investor_node, &Trust, 1, investor_monetary_action , first_action, true);
									shift_updates(&investor_node, &Trust, 1, investor_monetary_action);
									investor_shifts = Trust -> get_shifts();
									for(int irr=0; irr < noi; ++irr)
									{
										store_investor_shifts[time_step][1][irr] = investor_shifts[1][irr];
									}
								}
								investor_node->set_child(first_action, Trust);	//connect child node to tree
								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									//calculate action values for trustee simulated by investor (all possible trustees)
									guilt_parameter[noT]=guilt_counter;
									trustee_k(first_action
									, time_step
									, investor_planning
									, guilt_parameter
									, -1
									, hold_actions
									, hold_responses
									, investor_ui_init_system
									, investor_ut_init_system
									, investor_expected_trustee_payoff
									, investor_expected_outcome_utility
									, investor_temperature
									, mempool
									, &Trust);
								}
								if(!Trust-> expectation_set(0))	//determine expectation analogous to investor case
								{
									if((Subject_ToM[sub_id][0]+3)%2== 1)
									{
										double sum = 0.0;
										double isum = 0.0;
										investor_belief_parameters = Trust -> get_belief_parameters();
										investor_irr_beliefs = Trust -> get_irr_beliefs();
										for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
										{
											main_running_belief_probability[guilt_counter] = investor_belief_parameters[0][guilt_counter];
											sum += main_running_belief_probability[guilt_counter];
										}
										for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
										{
											main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
										}
										for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
										{
											main_irr_probability[guilt_counter] = investor_irr_beliefs[0][guilt_counter];
											isum += main_irr_probability[guilt_counter];
										}
										for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
										{
											main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
										}
										trustee_expectation_calculation( 0, &Trust,  main_running_belief_probability , main_irr_probability, first_action);
									}
									else
									{

										for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
										{
											main_running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
										}
										for(int irr_counter=0; irr_counter < noi; ++irr_counter)
										{
											main_irr_probability[irr_counter] = 0.0;
										}
										main_irr_probability[0]= 1.0;
										trustee_expectation_calculation( -1, &Trust,  main_running_belief_probability , main_irr_probability, first_action);
									}
								}
								//------------------------------ trustee -------------------------------
								Trust = new True_node();	//do the same as above for the trustee point of view
								trustee_irr_beliefs = trustee_node -> get_irr_beliefs();
								for(int level=0; level < noT; ++level)
								{
									for(int irr=0; irr < noi; ++irr)
									{
										Trust-> set_irr_parameters(irr, trustee_irr_beliefs[level][irr], level);
									}
								}
								trustee_node->set_child(first_action, Trust);
								trustee_belief_parameters = trustee_node-> get_belief_parameters();
								for(int level=0; level < noT; ++level)
								{
									for(int belief=0; belief < nob; ++belief)
									{
										Trust-> set_belief_parameters(belief, trustee_belief_parameters[level][belief], level);
									}
								}
								trustee_shifts = trustee_node -> get_shifts();
								for(int level=0; level < noT+1; ++level)
								{
									for(int irr=0; irr < noi; ++irr)
									{
										Trust -> set_shift(level, trustee_shifts[level][irr], irr);
									}
								}
								//trustee_node->set_child(first_action, Trust);
								if((Subject_ToM[sub_id][1]+3)%2== 0 )
								{
									shift_updates(&trustee_node, &Trust, 0, investor_monetary_action);
									trustee_shifts = Trust -> get_shifts();
									for(int irr=0; irr < noi; ++irr)
									{
										store_trustee_shifts[time_step][0][irr] = trustee_shifts[0][irr];
									}
								}
								if((Subject_ToM[sub_id][1]+3)%2== 1 )
								{
									//belief_updates(&trustee_node, &Trust, 1, investor_monetary_action, first_action, true);
									belief_updates(&trustee_node, &Trust, 1, investor_monetary_action, first_action, true);
									shift_updates(&trustee_node, &Trust, 1, investor_monetary_action);
									trustee_shifts = Trust -> get_shifts();
									for(int irr=0; irr < noi; ++irr)
									{
										store_trustee_shifts[time_step][1][irr] = trustee_shifts[1][irr];
									}
								}
								for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
								{
									guilt_parameter[noT]=guilt_counter;
									trustee_k(first_action
									, time_step
									, trustee_planning
									, guilt_parameter
									, -1
									//, trustee_belief_parameters
									, hold_actions
									, hold_responses
									, trustee_ui_init_system
									, trustee_ut_init_system
									, trustee_expected_trustee_payoff
									, trustee_expected_outcome_utility
									, trustee_temperature
									, mempool
									, &Trust);

								}
								if(!Trust-> expectation_set(0))
								{
									if((Subject_ToM[sub_id][1]+3)%2== 0)
									{ 		//determine probabilites from Dirichlet-Multinomial priors
										double sum = 0.0;
										double isum = 0.0;
										trustee_belief_parameters = Trust -> get_belief_parameters();
										trustee_irr_beliefs = Trust -> get_irr_beliefs();
										for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
										{
											main_running_belief_probability[guilt_counter] = trustee_belief_parameters[0][guilt_counter];
											sum += main_running_belief_probability[guilt_counter];
										}
										for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
										{
											main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
										}
										for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
										{
											main_irr_probability[guilt_counter] = trustee_irr_beliefs[0][guilt_counter];
											isum += main_irr_probability[guilt_counter];
										}
										for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
										{
											main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
										}
										//Calculate expectation
										trustee_expectation_calculation( 0, &Trust,  main_running_belief_probability , main_irr_probability, first_action);
										store_trustee_expectation[time_step][0] = trustee_node ->get_expectation(0);
									}
									else
									{
										//level -1 has uniform probabilites
										for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
										{
											main_running_belief_probability[guilt_counter] = 1.0/static_cast<double>(nob);
										}
										for(int irr_counter=0; irr_counter < noi; ++irr_counter)
										{
											main_irr_probability[irr_counter] = 0.0;
										}
										main_irr_probability[0]= 1.0;
										trustee_expectation_calculation( -1, &Trust,  main_running_belief_probability , main_irr_probability, first_action);
										store_trustee_expectation[time_step][0] = trustee_node ->get_expectation(0);
									}
								}
							}
							//-------------------------------------------- End Level -1 set-up ----------------------------------
							//-------------------------------------------- Investor Choice Begin --------------------------------
							for(int level = 1; level <= Subject_ToM[sub_id][0]+1 ; ++level) //all initial investor considerations
							{
								if((Subject_ToM[sub_id][0]+3-level)%2== 0)
								{
									for(int belief=0; belief < nob ; ++belief)
									{
										//calculate investor choice probability if not already calculated
										if( !(investor_node-> get_confirm_exploration(belief, level)) )
										{
											guilt_parameter[level-1]=belief;
											investor_k( time_step
											, investor_planning
											, guilt_parameter
											, level-1
											//, investor_belief_parameters
											, Subject_Games[sub_id][0]
											, Subject_Games[sub_id][1]
											, investor_ui_init_system
											, investor_ut_init_system
											, investor_expected_trustee_payoff
											, investor_expected_outcome_utility
											, investor_temperature
											, mempool
											, &investor_node);
										}


									}
									//calculate expectation of investor action, if not already calculated
									if(level <(Subject_ToM[sub_id][0]+1))
									{
										if(!investor_node -> expectation_set(level))
										{
											double sum = 0.0;
											double isum = 0.0;
											investor_belief_parameters = investor_node -> get_belief_parameters();
											investor_irr_beliefs = investor_node -> get_irr_beliefs();
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
											{
												main_running_belief_probability[guilt_counter] = investor_belief_parameters[level][guilt_counter];
												sum += main_running_belief_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
											{
												main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = investor_irr_beliefs[level][guilt_counter];
												isum += main_irr_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
											}
											investor_expectation_calculation( level, &investor_node,  main_running_belief_probability , main_irr_probability);
										}
										store_investor_expectation[time_step][level] = investor_node->get_expectation(level);
									}

								}

							}

							for(int level = 0; level <= Subject_ToM[sub_id][0]; ++level)
							{
								if((Subject_ToM[sub_id][0]+level+2)%2== 0)
								{
									//extract all relevant choice probabilities. Irritated and nonirritated
									//stoe them for later likelihood calculation
									main_exp_payoffs = investor_node->get_exp_payoffs(level+1);
									store_investor_payoffs[time_step][level] = main_exp_payoffs;
									main_i_exp_payoffs = Irritation_Investor ->get_exp_payoffs(0);
									store_investor_shifted_payoffs[time_step][level] = main_i_exp_payoffs;
								}
							}

							for(int level = 1; level <= Subject_ToM[sub_id][1]+1 ; ++level)
							{
								if((Subject_ToM[sub_id][1]+3-level)%2== 1)
								{
									for(int belief=0; belief < nob ; ++belief)
									{
										//calculate trustee model of investor choice probabilities
										if(!(trustee_node-> get_confirm_exploration(belief, level)))
										{
											guilt_parameter[level-1]=belief;
											investor_k( time_step
											, trustee_planning
											, guilt_parameter
											, level-1
											, Subject_Games[sub_id][0]
											, Subject_Games[sub_id][1]
											, trustee_ui_init_system
											, trustee_ut_init_system
											, trustee_expected_trustee_payoff
											, trustee_expected_outcome_utility
											, trustee_temperature
											, mempool
											, &trustee_node);

										}


									}
									// set expectation of investor action, if not already set
									if(level <(Subject_ToM[sub_id][1]+1))
									{
										if(!trustee_node -> expectation_set(level))
										{
											double sum = 0.0;
											double isum = 0.0;
											trustee_belief_parameters = trustee_node -> get_belief_parameters();
											trustee_irr_beliefs = trustee_node -> get_irr_beliefs();
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
											{
												main_running_belief_probability[guilt_counter] = trustee_belief_parameters[level][guilt_counter];
												sum += main_running_belief_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
											{
												main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = trustee_irr_beliefs[level][guilt_counter];
												isum += main_irr_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
											}
											investor_expectation_calculation( level, &trustee_node,  main_running_belief_probability , main_irr_probability);
										}
									}
									store_trustee_expectation[time_step][level] = trustee_node ->get_expectation(level);
								}


							}
							// build actual choice probability from mixture of irritated and non-irritated choice
							for(int level = 0; level < Subject_ToM[sub_id][0]+1; ++level)
							{
								if((Subject_ToM[sub_id][0]+level+2)%2== 0)
								{
									investor_payoffs = store_investor_payoffs[time_step][level];
									shifted_payoffs = store_investor_shifted_payoffs[time_step][level];
									investor_shifts = investor_node -> get_shifts();
									for(int bel=0; bel < nob; ++bel)
									{
										for(int irr =0; irr < noi; ++irr)
										{
											for(int p_act =0; p_act < noa; ++p_act)
											{
												if((1.0-investor_shifts[level+1][irr]) > 0.001 & investor_payoffs[bel][p_act]>0.0001)
												{
													hold_investor_payoffs[level][time_step][irr][bel][p_act] = (1.0-investor_shifts[level+1][irr])*(
												investor_payoffs[bel][p_act]);
												}
												else
												{
													hold_investor_payoffs[level][time_step][irr][bel][p_act] = 0.0;
												}
												if((investor_shifts[level+1][irr])>0.001 & shifted_payoffs[0][p_act]>0.001)
												{
													hold_investor_payoffs[level][time_step][irr][bel][p_act]+= (investor_shifts[level+1][irr])*shifted_payoffs[0][p_act];
												}

											}
											//---------------------------- likelihood update block -----------------------------------------------
											if(hold_investor_payoffs[level][time_step][irr][bel][Subject_Games[sub_id][0][time_step]]> 0.000001)
											{
												like[irr][level][bel] += -log(hold_investor_payoffs[level][time_step][irr][bel][Subject_Games[sub_id][0][time_step]]);
											}
											else
											{
												like[irr][level][bel] += 1000.0;
											}
										}
									}
								}
							}
							//---------------------------------------- Generative Block ----------------------------------------------
							//Subject_Games[sub_id][0][time_step] = softmax( hold_investor_payoffs[Subject_ToM[sub_id][0]][time_step][investor_irr][Subject_Guilt[sub_id][0]], 201202);
							//ofs.write( reinterpret_cast<char*>( &Subject_Games[sub_id][0][time_step] ), sizeof Subject_Games[sub_id][0][time_step]);
							//--------------------------------------------------------------------------------------------------------
							double investor_monetary_action = 5.0*static_cast<double>(Subject_Games[sub_id][0][time_step]);


							temp_node = trustee_node->get_child(Subject_Games[sub_id][0][time_step]);	//move to next history node, according to investor action
							shift_updates(&trustee_node, &temp_node, Subject_ToM[sub_id][1]+1, investor_monetary_action);							
							//-------------------------- Trustee Choice calculation block -------------------------------------------------------------------
							for(int level =1; level <= Subject_ToM[sub_id][1]+1 ; ++level)
							{
								trustee_belief_parameters = trustee_node -> get_belief_parameters();
								trustee_irr_beliefs = trustee_node -> get_irr_beliefs();
								trustee_shifts = trustee_node -> get_shifts();
								for(int b=0; b < nob; ++b)
								{
									temp_node-> set_belief_parameters(b, trustee_belief_parameters[level-1][b], level-1);
								}
								for(int irr=0; irr < noi; ++irr)
								{
									temp_node-> set_irr_parameters(irr, trustee_irr_beliefs[level-1][irr], level-1);
								}

								for(int irr=0; irr < noi; ++irr)
								{
									//temp_node -> set_shift(level-1, trustee_shifts[level-1][irr], irr);
									temp_node -> set_shift(level, trustee_shifts[level][irr], irr);
								}
								if((Subject_ToM[sub_id][1]+3-level)%2== 0)
								{
									
									//belief updating on investor action
									belief_updates(&trustee_node, &temp_node, level, investor_monetary_action, Subject_Games[sub_id][0][time_step], true);
									//possible irritation on investor action
									shift_updates(&trustee_node, &temp_node, level, investor_monetary_action);
									trustee_shifts = temp_node -> get_shifts();
									for(int irr=0; irr < noi; ++irr)
									{
										store_trustee_shifts[time_step][level][irr] = trustee_shifts[level][irr];
									}
									for(int belief=0; belief < nob ; ++belief)
									{
										//---------------------- action probability calculation, if not already explored -----------------
										if(!(temp_node-> get_confirm_exploration(belief, level)))
										{
											guilt_parameter[level-1]=belief;
											trustee_k( Subject_Games[sub_id][0][time_step]
											, time_step
											, trustee_planning
											, guilt_parameter
											, level-1
											, Subject_Games[sub_id][0]
											, Subject_Games[sub_id][1]
											, trustee_ui_init_system
											, trustee_ut_init_system
											, trustee_expected_trustee_payoff
											, trustee_expected_outcome_utility
											, trustee_temperature
											, mempool
											, &temp_node);
										}
									}
									//------------------------------ expectation on trustee action, if not yet calculated -------------------------------------------
									if(level <(Subject_ToM[sub_id][1]+1))
									{
										if(!temp_node -> expectation_set(level))
										{
											double sum = 0.0;
											double isum = 0.0;
											trustee_belief_parameters = temp_node -> get_belief_parameters();
											trustee_irr_beliefs = temp_node -> get_irr_beliefs();
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
											{
												main_running_belief_probability[guilt_counter] = trustee_belief_parameters[level][guilt_counter];
												sum += main_running_belief_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
											{
												main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = trustee_irr_beliefs[level][guilt_counter];
												isum += main_irr_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
											}
											trustee_expectation_calculation( level, &temp_node,  main_running_belief_probability , main_irr_probability, Subject_Games[sub_id][0][time_step]);
										}

									}
									store_trustee_expectation[time_step][level] = temp_node ->get_expectation(level);
								}


							}

							if((Subject_Games[sub_id][0][time_step]==0))
							{
								//no action required by trustee, if no investment took place
							}
							if(!(Subject_Games[sub_id][0][time_step]==0))
							{
								temp_node = trustee_node -> get_child(Subject_Games[sub_id][0][time_step]);	//move to correct sub-node along investor action
								for(int level = 0; level < Subject_ToM[sub_id][1]+1; ++level)
								{
									if((Subject_ToM[sub_id][1]+level+2)%2== 0)
									{	//get action probabilities for all relevant levels
										main_exp_payoffs = temp_node->get_exp_payoffs(level+1);
										main_i_exp_payoffs = temp_node->get_exp_payoffs(0);
										//store relevant action probabilities for potential output
										store_trustee_payoffs[time_step][level] = main_exp_payoffs;
										store_trustee_shifted_payoffs[time_step][level] = main_i_exp_payoffs;

									}
								}

							}

							if(!(Subject_Games[sub_id][0][time_step]==0))
							{
								for(int level = 0; level < Subject_ToM[sub_id][1]+1; ++level)
								{
									if( (Subject_ToM[sub_id][1]+level+2)%2== 0)
									{
										//use earlier extracted probabilities
										trustee_payoffs = store_trustee_payoffs[time_step][level];
										shifted_payoffs = store_trustee_shifted_payoffs[time_step][level];
										trustee_shifts = temp_node->get_shifts();
										// form current probabilities by mixing appropriate irritated and non-irritated values
										// test all possible settings for guilt, irritation and level
										for(int bel=0; bel < nob; ++bel)
										{
											for(int irr =0; irr < noi; ++irr)
											{
												for(int pot_act =0; pot_act  < noa; ++pot_act )
												{

													if((1.0-trustee_shifts[level+1][irr])>0.001 & trustee_payoffs[bel][pot_act]>0.001)
													{
														hold_trustee_payoffs[level][time_step][irr][bel][pot_act]= (1.0-trustee_shifts[level+1][irr])*(
														trustee_payoffs[bel][pot_act]);
													}
													else
													{
														hold_trustee_payoffs[level][time_step][irr][bel][pot_act] = 0.0;
													}
													if(trustee_shifts[level+1][irr]>0.001 & shifted_payoffs[0][pot_act]> 0.00001)
													{
														hold_trustee_payoffs[level][time_step][irr][bel][pot_act]+= (trustee_shifts[level+1][irr])*shifted_payoffs[0][pot_act];
													}
												}
												if(hold_trustee_payoffs[level][time_step][irr][bel][Subject_Games[sub_id][1][time_step]] > 0.00001)
												{	//build up appropriate negative-log-likelihoods at every time step
													Tlike[irr][level][bel] += -log(hold_trustee_payoffs[level][time_step][irr][bel][Subject_Games[sub_id][1][time_step]]);
												}
												else
												{
													Tlike[irr][level][bel] += 1000.0;
												}
											}
										}
									}
									else
									{
										for(int irr =0; irr < noi; ++irr)
										{
											for(int bel=0; bel < nob; ++bel)
											{
												for(int p_act =0; p_act  < noa; ++p_act )
												{
													hold_trustee_payoffs[level][time_step][irr][bel][p_act] = 0.0;
												}
											}
										}
									}
								}

							}
							else
							{
								for(int level = 0; level < noT; ++level)
								{
									for(int irr =0; irr < noi; ++irr)
									{
										for(int bel=0; bel < nob; ++bel)
										{
											for(int p_act =0; p_act  < noa; ++p_act )
											{
												hold_trustee_payoffs[level][time_step][irr][bel][p_act] = 0.0;
											}
										}
									}
								}
							}

							trustee_belief_parameters = temp_node -> get_belief_parameters();
							trustee_shifts = temp_node -> get_shifts();
							trustee_irr_beliefs = temp_node -> get_irr_beliefs();
							for(int k =0; k < noT; ++k)
							{
								for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
								{
									store_trustee_belief_parameters[time_step][k][guilt_belief] = trustee_belief_parameters[k][guilt_belief];
								}
								for(int irr_belief=0; irr_belief < noi; ++irr_belief)
								{
									store_trustee_irr_beliefs[time_step][k][irr_belief] = trustee_irr_beliefs[k][irr_belief];
								}
							}
							//------------------------------Generative Trustee Block
							/*if(!(Subject_Games[sub_id][0][time_step]==0))
							{
								Subject_Games[sub_id][1][time_step] = softmax( hold_trustee_payoffs[Subject_ToM[sub_id][1]][time_step][trustee_irr][Subject_Guilt[sub_id][1]], 201202);
							}
							else
							{
								Subject_Games[sub_id][1][time_step] = 0;
							}*/
							//ofs.write( reinterpret_cast<char*>( &Subject_Games[sub_id][1][time_step] ), sizeof Subject_Games[sub_id][1][time_step] );
							// Investor Inference
							double trustee_monetary_response = 1.0/6.0*static_cast<double>(rbf*5*Subject_Games[sub_id][0][time_step]*Subject_Games[sub_id][1][time_step]);

							temp_node = investor_node->get_child(Subject_Games[sub_id][0][time_step]); //move from root to trustee response along the chosen action
							//calculate the trustee response likelihoods
							for(int level = 1; level <=Subject_ToM[sub_id][0]+1 ; ++level)
							{
								investor_belief_parameters = investor_node -> get_belief_parameters();
								investor_irr_beliefs = investor_node -> get_irr_beliefs();
								investor_shifts = investor_node -> get_shifts();
								for(int b=0; b < nob; ++b)
								{
									temp_node-> set_belief_parameters(b, investor_belief_parameters[level-1][b], level-1);
								}
								for(int irr=0; irr < noi; ++irr)
								{
									temp_node-> set_irr_parameters(irr, investor_irr_beliefs[level-1][irr], level-1);
								}

								for(int irr=0; irr < noi; ++irr)
								{
									//temp_node -> set_shift(level-1, investor_shifts[level-1][irr], irr);
									temp_node -> set_shift(level, investor_shifts[level][irr], irr);
								}

								if((Subject_ToM[sub_id][0]+3-level)%2== 1)
								{
									// belief and shift updates
									belief_updates(&investor_node, &temp_node, level, investor_monetary_action, Subject_Games[sub_id][0][time_step], true);
									shift_updates(&investor_node, &temp_node, level, investor_monetary_action);
									//----
									
									investor_shifts = temp_node -> get_shifts();
									for(int irr=0; irr < noi; ++irr)
									{
										store_investor_shifts[time_step][level][irr] = investor_shifts[level][irr];
									}
									for(int guilt=0; guilt < nob; ++guilt)
									{
										//calculate action probabilites, unless already explored
										if(!(temp_node->get_confirm_exploration(guilt, level)))
										{

											//	}
											//}
											guilt_parameter[level-1]=guilt;
											trustee_k(Subject_Games[sub_id][0][time_step]
											, time_step
											, investor_planning
											, guilt_parameter
											, level-1
											, Subject_Games[sub_id][0]
											, Subject_Games[sub_id][1]
											, investor_ui_init_system
											, investor_ut_init_system
											, investor_expected_trustee_payoff
											, investor_expected_outcome_utility
											, investor_temperature
											, mempool
											, &temp_node);
										}
									}
									//------------------------------- set expectation sub-block -----------------------
									if(level <(Subject_ToM[sub_id][0]+1))
									{
										if(!temp_node -> expectation_set(level))
										{
											double sum = 0.0;
											double isum = 0.0;
											investor_belief_parameters = temp_node -> get_belief_parameters();
											investor_irr_beliefs = temp_node -> get_irr_beliefs();
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter) //level needs to be -1 here
											{
												main_running_belief_probability[guilt_counter] = investor_belief_parameters[level][guilt_counter];
												sum += main_running_belief_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < nob; ++guilt_counter)
											{
												main_running_belief_probability[guilt_counter] = main_running_belief_probability[guilt_counter]/sum;
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = investor_irr_beliefs[level][guilt_counter];
												isum += main_irr_probability[guilt_counter];
											}
											for(int guilt_counter=0; guilt_counter < noi; ++guilt_counter)
											{
												main_irr_probability[guilt_counter] = main_irr_probability[guilt_counter]/isum;
											}
											trustee_expectation_calculation( level, &temp_node,  main_running_belief_probability , main_irr_probability, Subject_Games[sub_id][0][time_step]);
										}

									}
									store_investor_expectation[time_step][level] = temp_node->get_expectation(level);
								}

							}
							//get investor final beliefs and shifts
							temp_node = investor_node->get_child(Subject_Games[sub_id][0][time_step]);
							store_investor_expectation[time_step][0] = temp_node->get_expectation(0);
							Trust = temp_node-> get_child(Subject_Games[sub_id][1][time_step]);
							if(Trust) //save beliefs and shifts outside of the node, so at the beginning of the next round, nodes can be initialized to current values
							{
								if( !(Subject_Games[sub_id][0][time_step]==0))
								{							
									shift_updates(&temp_node, &Trust, Subject_ToM[sub_id][0]+1 , trustee_monetary_response);
									if(Subject_Games[sub_id][0][time_step]>1)
									{
										belief_updates(&temp_node, &Trust, Subject_ToM[sub_id][0]+1 , trustee_monetary_response, Subject_Games[sub_id][1][time_step], false);
									}
								}									
								investor_belief_parameters = Trust -> get_belief_parameters();
								investor_shifts = Trust -> get_shifts();
								investor_irr_beliefs = Trust -> get_irr_beliefs();
								for(int k =0; k < noT; ++k)
								{
									for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
									{
										store_investor_belief_parameters[time_step][k][guilt_belief] = investor_belief_parameters[k][guilt_belief];
									}
									for(int irr_belief=0; irr_belief < noi; ++irr_belief)
									{
										store_investor_irr_beliefs[time_step][k][irr_belief] = investor_irr_beliefs[k][irr_belief];
									}
								}
							}
							//get trustee final beliefs and shifts
							temp_node = trustee_node->get_child(Subject_Games[sub_id][0][time_step]);
							Trust = temp_node ->get_child(Subject_Games[sub_id][1][time_step]);

							if(Trust)	//save beliefs and shifts outside of the node, so at the beginning of the next round, nodes can be initialized to current values
							{										
								trustee_belief_parameters = Trust -> get_belief_parameters();
								trustee_shifts = Trust -> get_shifts();
								trustee_irr_beliefs = Trust -> get_irr_beliefs();
							}

							delete investor_node; //delete current calculation investor

							delete trustee_node; //delete current calculation trustee

							mempool.DeleteAll(); //clear local mempool

						}

						for(int level = 0; level < noT; ++level)
						{
							
								for(int irr =0; irr < 1; ++irr)
								{
									if(level <= Subject_ToM[sub_id][0]){
									//----------------------------- Investor winning model block ----------------------------------
									if((Subject_ToM[sub_id][0]+level+2)%2== 0)	//possible investor levels in steps of 2 from the specified level (Subject_ToM)
									{
										for(int val_Guilt = 0 ; val_Guilt < nob; ++val_Guilt)
										{
											if(like[irr][level][val_Guilt] < Ilike_max)
											{ //---------------------  Update winning parameters for later output, if Negative-log-likelihood has improved ------------
												WI_ToM = level;
												WI_G =val_Guilt;
												Ilike_max = like[irr][level][val_Guilt];
												WI_risk = risk_count;
												WI_irritability = irr;
												WI_shift = irr_belief;
												WI_P = d1;
												WI_temp = investor_temperature;
												for(int time_step =0; time_step < global_time_horizon; ++time_step)
												{
													for(int k =0; k < noT; ++k)
													{
														WI_investor_expectation[time_step][k] = store_investor_expectation[time_step][k];
														for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
														{
															WI_investor_belief_parameters[time_step][k][guilt_belief] = store_investor_belief_parameters[time_step][k][guilt_belief];
														}
														for(int irr_new=0; irr_new < noi; ++irr_new)
														{
															WI_irr_beliefs[time_step][k][irr_new]=store_investor_irr_beliefs[time_step][k][irr_new];
															WI_investor_shifts[time_step][k][irr_new] = store_investor_shifts[time_step][k][irr_new];
															WI_investor_shifts[time_step][k+1][irr_new] = store_investor_shifts[time_step][k+1][irr_new];
														}
														for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
														{
															for(int irr_new=0; irr_new < noi; ++irr_new)
															{
																for(int p_act =0; p_act  < noa; ++p_act )
																{
																	WI_investor_payoffs[k][time_step][irr_new][guilt_belief][p_act] = hold_investor_payoffs[k][time_step][irr_new][guilt_belief][p_act];
																}
															}
														}
													}
												}
											}

										}
									}
								}
								//----------------------------- Trustee winning model block ----------------------------------
								if(level <= Subject_ToM[sub_id][1]){
								if((Subject_ToM[sub_id][1]+level+2)%2== 0)
								{
									for(int val_Guilt = 0 ; val_Guilt < nob; ++val_Guilt)
									{

										if(Tlike[irr][level][val_Guilt] < Tlike_max)
										{	//---------------------  Update winning parameters for later output, if Negative-log-likelihood has improved ------------
											WT_temp = trustee_temperature;
											WT_ToM = level;
											WT_G = val_Guilt;
											WT_risk = risk_count;
											WT_irritability = irr;
											WT_shift = irr_belief;

											Tlike_max = Tlike[irr][level][val_Guilt];
											WT_P = d2;
											for(int time_step =0; time_step < global_time_horizon; ++time_step)
											{

												for(int k =0; k < noT; ++k)
												{
													WT_trustee_expectation[time_step][k] = store_trustee_expectation[time_step][k];
													for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
													{
														WT_trustee_belief_parameters[time_step][k][guilt_belief] = store_trustee_belief_parameters[time_step][k][guilt_belief];
													}
													for(int irr_new=0; irr_new < noi; ++irr_new)
													{
														WT_irr_beliefs[time_step][k][irr_new]=store_trustee_irr_beliefs[time_step][k][irr_new];
														WT_trustee_shifts[time_step][k][irr_new] = store_trustee_shifts[time_step][k][irr_new];
														WT_trustee_shifts[time_step][k+1][irr_new] = store_trustee_shifts[time_step][k+1][irr_new];
													}
													for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
													{
														for(int irr_new=0; irr_new < noi; ++irr_new)
														{
															for(int p_act =0; p_act  < noa; ++p_act )
															{
																WT_trustee_payoffs[k][time_step][irr_new][guilt_belief][p_act] = hold_trustee_payoffs[k][time_step][irr_new][guilt_belief][p_act];
															}
														}
													}
												}
											}
										}

									}
								}
								}
							}
						}
					}
				}
			}
		}

		// ---------------------------------------------------------- Output block ----------------------------------------------------------

		ofs.write( reinterpret_cast<char*>( &Ilike_max ), sizeof Ilike_max );

		ofs.write( reinterpret_cast<char*>( &WI_ToM ), sizeof WI_ToM );

		ofs.write( reinterpret_cast<char*>( &WI_G ), sizeof WI_G );

		ofs.write( reinterpret_cast<char*>( &WI_risk ), sizeof WI_risk );

		ofs.write( reinterpret_cast<char*>( &WI_P ), sizeof WI_P );

		ofs.write( reinterpret_cast<char*>( &WI_temp ), sizeof WI_temp );

		ofs.write( reinterpret_cast<char*>( &WI_shift ), sizeof WI_shift);

		ofs.write( reinterpret_cast<char*>( &WI_irritability ), sizeof WI_irritability );

		ofs.write( reinterpret_cast<char*>( &Tlike_max ), sizeof Tlike_max );

		ofs.write( reinterpret_cast<char*>( &WT_ToM ), sizeof WT_ToM );

		ofs.write( reinterpret_cast<char*>( &WT_G ), sizeof WT_G );

		ofs.write( reinterpret_cast<char*>( &WT_risk ), sizeof WT_risk );

		ofs.write( reinterpret_cast<char*>( &WT_P ), sizeof WT_P );

		ofs.write( reinterpret_cast<char*>( &WT_temp ), sizeof WT_temp );

		ofs.write( reinterpret_cast<char*>( &WT_shift ), sizeof WT_shift);

		ofs.write( reinterpret_cast<char*>( &WT_irritability ), sizeof WT_irritability);
		//ofs.write( reinterpret_cast<char*>( &WT_irritability ), sizeof WT_irritability);

		for(int time_step =0; time_step < global_time_horizon; ++time_step)
		{
			for(int k =0; k < noT; ++k)
			{
				ofs.write( reinterpret_cast<char*>( &WI_investor_expectation[time_step][k] ), sizeof WI_investor_expectation[time_step][k]);
				for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
				{
					ofs.write( reinterpret_cast<char*>(&WI_investor_belief_parameters[time_step][k][guilt_belief]), sizeof WI_investor_belief_parameters[time_step][k][guilt_belief]);
				}
				for(int irr_belief=0; irr_belief < noi; ++irr_belief)
				{
					ofs.write( reinterpret_cast<char*>(&WI_irr_beliefs[time_step][k][irr_belief]), sizeof WI_irr_beliefs[time_step][k][irr_belief]);
					ofs.write( reinterpret_cast<char*>(&WI_investor_shifts[time_step][k][irr_belief]), sizeof WI_investor_shifts[time_step][k][irr_belief]);
					ofs.write( reinterpret_cast<char*>(&WI_investor_shifts[time_step][k+1][irr_belief]), sizeof WI_investor_shifts[time_step][k+1][irr_belief]);
				}
				for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
				{
					for(int irr_belief=0; irr_belief < noi; ++irr_belief)
					{
						for(int p_act =0; p_act  < noa; ++p_act )
						{
							ofs.write( reinterpret_cast<char*>(&WI_investor_payoffs[k][time_step][irr_belief][guilt_belief][p_act]), sizeof WI_investor_payoffs[k][time_step][irr_belief][guilt_belief][p_act]);
						}
					}
				}
			}
		}
		for(int time_step =0; time_step < global_time_horizon; ++time_step)
		{
			for(int k =0; k < noT; ++k)
			{
				ofs.write( reinterpret_cast<char*>( &WT_trustee_expectation[time_step][k] ), sizeof WT_trustee_expectation[time_step][k]);
				for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
				{
					ofs.write( reinterpret_cast<char*>(&WT_trustee_belief_parameters[time_step][k][guilt_belief]), sizeof WT_trustee_belief_parameters[time_step][k][guilt_belief]);
				}
				for(int irr_belief=0; irr_belief < noi; ++irr_belief)
				{
					ofs.write( reinterpret_cast<char*>(&WT_irr_beliefs[time_step][k][irr_belief]), sizeof WT_irr_beliefs[time_step][k][irr_belief]);
					ofs.write( reinterpret_cast<char*>(&WT_trustee_shifts[time_step][k][irr_belief]), sizeof WT_trustee_shifts[time_step][k][irr_belief]);
					ofs.write( reinterpret_cast<char*>(&WT_trustee_shifts[time_step][k+1][irr_belief]), sizeof WT_trustee_shifts[time_step][k+1][irr_belief]);
				}
				for(int guilt_belief=0; guilt_belief < nob; ++guilt_belief)
				{
					for(int irr_belief=0; irr_belief < noi; ++irr_belief)
					{
						for(int p_act =0; p_act  < noa; ++p_act )
						{
							ofs.write( reinterpret_cast<char*>(&WT_trustee_payoffs[k][time_step][irr_belief][guilt_belief][p_act]), sizeof WT_trustee_payoffs[k][time_step][irr_belief][guilt_belief][p_act]);
						}
					}
				}
			}
		}
	}
	ofs.close();
	delete Irritation_Investor;
	mexpool.DeleteAll();
}
