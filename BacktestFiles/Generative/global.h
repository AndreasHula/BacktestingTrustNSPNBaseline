#ifndef GLOBAL_H_INCLUDE_GUARD
#define GLOBAL_H_INCLUDE_GUARD

#include <ostream>
#include <iostream>
#include <vector>
#include <utility>
#include <iterator>
#include <fstream>
#include <array>

#include <math.h>
#include <time.h>
#include <deque>
#include <random>

#include <boost/function.hpp>
#include <boost/utility.hpp>



#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>


#include "memorypool.h"



using namespace std;

typedef boost::numeric::ublas::matrix<double> Matrix;
typedef boost::numeric::ublas::matrix_row<const Matrix> Matrix_row;
typedef boost::numeric::ublas::matrix_column<const Matrix> Matrix_column;
typedef boost::numeric::ublas::zero_matrix<double> Zero_matrix;

typedef std::vector<Matrix> Matrix_vector;
typedef std::vector<Matrix::size_type> Index_vector;

const int max_amount = 4; // how much the investor can offer up (and is thus at risk)
const int global_time_horizon = 10; //number of exchanges between investor and trustee
const int nob = 3; //number of believes
const Matrix::size_type noa = max_amount + 1; //number of actions
const int ActionResponsePairs = 21;
const int max_plan = 5;
const int noi = 5; //Maximum number of irritation values
const int noT = 5; //Maximum number of ToM levels-1 (level 0 counts as one level)
//const int nav = 8;//Number of risk aversion factors


//game specifier
const int rbf =3; // risk benefit factor (the number that the offered amount is multiplied by)
//const double temperature =3.0; //softmax inverse temperature parameter
//simulation specifier
const int simulation_iterations =2000; //how often the iterated game is simulated
const double exploration_constant = 2.0; //bonus in exploration term
const double eps =0.5;




int pmax(int value);

std::array<int, global_time_horizon-1> custom_sort(std::array<int, global_time_horizon-1> hist, int time);

void print_matrix(Matrix const & m);
std::vector<double> greedy_probabilities(unsigned int preferred_action, unsigned int actions);

template <class T> //for array
int softmax(const T& probabilities, int const& Ind)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
	double draw = dis(gen);//generate_uniform_distributed_value();
	int choice = 0;

	std::vector<double> new_prob (probabilities.size(), 0.0);	
	//include normalisation loop 
	double norm= 0.0;
	for(int i=0; i < probabilities.size() ; ++i)
	{
		norm += probabilities[i];
	}
	
	for(int i=0; i < probabilities.size() ; ++i)
	{
		new_prob[i] = probabilities[i]/norm;
	}
	norm= 0.0;
	for(int i=0; i < probabilities.size() ; ++i)
	{
		norm += new_prob[i];
	}	

	double sum = new_prob[0];	
	while(sum < draw)
	{
		++choice;
		if(choice >= probabilities.size())
		{
			cout << "Error in softmax: choice: " << choice << " size: " << probabilities.size() << "Index" << Ind << endl;
			cout << "sums to" << norm << endl;			
			for(int i =0; i < probabilities.size(); ++i)
			{
				cout << probabilities[i] << endl;
			}
			for(int i =0; i < probabilities.size(); ++i)
			{
				cout << new_prob[i] << endl;
			}			
			cout << "Draw" << draw << endl;			
			assert(false);
		}
		sum += new_prob[choice];
	}

	return choice;
}

inline int nor(int action)
{
	return (action == 0 ? 1 : noa);
}

inline int minimalize(double value, std::array<double,noa> reference)
{
	int min_int=0;
	double distance =40.0;
	for(int i=0; i <noa; ++i)
	{
		if(abs(value - reference[i]) < distance )
		{
			min_int = i;
			distance = abs(value - reference[i]);
		}
	}

	return min_int;
}


inline int math_round(double val)
{
	return static_cast<int>(val + 0.5);
}

template<class T>
void print_vector(T const& v)
{
	for(unsigned int i = 0; i < v.size(); ++i)
	{
		cout << i << ": " << v[i] << "\n";
	}
	cout << endl;
}

template <class T>
void greedy_probabilities(T begin, T end, unsigned int preferred_action)
{
	unsigned int actions = distance(begin, end);

	if(actions > 1)
	{
		for(unsigned int i = 0; i < actions; ++i, ++begin)
		{
			*begin = (i == preferred_action ? 1-eps : eps/(actions-1));
		}
	}
	else
	{
		*begin = 1.0;
	}
}

class True_node : public MEMORY_OBJECT
{
	public:
		True_node()//				  //m_free_child(0),				  //m_free_counter(0)
		{
			for(int l= 0; l < noT; ++l)
			{
				m_expectation[l] = 0.0;
				m_expectation_exists[l]= false;
				for(int b=0; b < nob; ++b)
				{
					m_sufficient_exploration[l][b] = false;
					//m_rolled_out[l][b] = false ;
					m_belief_parameters[l][b] = 0.0;
					//m_belief_probabilities[l][b] = 0.0;
				}
				for(int b=0; b < noi; ++b)
				{
					m_shifts[l][b]=0.0;
					m_shifts[l+1][b]=0.0;
					m_irr_belief[l][b] = 0.0;
					//m_irr_probabilities[l][b] =0.0;
				}
			}
			m_expectation[noT] = 0.0;
			m_expectation_exists[noT]= false;
			for(int b=0; b < nob; ++b)
			{
				m_sufficient_exploration[noT][b] = false;
				//m_rolled_out[noT][b] = false ;
			}
			for(int i = 0; i < noa; ++i)
			{
				m_children[i] =NULL;
				m_payoff[i] = 0.0;
				for(int l=0; l < (noT+1); ++l)
				{
					for(int b=0; b < nob; ++b)
					{
						m_exp_payoffs[l][b][i] = 0.0;
					}
				}
				//m_action_count[i] = 0;
			}

		}


		True_node* get_child(int action) const
		{
			return m_children[action];
		}

		void set_belief_parameters(int trustee_belief, double value,   int level)
		{
			m_belief_parameters[level][trustee_belief] = value;
		}

		void set_irr_parameters(int irr, double value,   int level)
		{
			m_irr_belief[level][irr] = value;
		}

		/*void set_irr_probabilities(int irr, double value,   int level)
		{
			m_irr_probabilities[level][irr] = value;
		}

		void set_belief_probabilities(int trustee_belief, double value,  int level)
		{
			m_belief_probabilities[level][trustee_belief] = value;
		}*/

		void set_exp_payoffs(int action, double value, int trustee_belief, int level)
		{
			m_exp_payoffs[level][trustee_belief][action] = value;
		}


		std::array<std::array<double, noa>, nob> get_exp_payoffs(int level) const
		{
			return m_exp_payoffs[level];
		}

		std::array<std::array<double,nob>, noT> get_belief_parameters() const
		{
			return m_belief_parameters;
		}

		/*std::array<double,nob> get_belief_probabilities(int level) const
		{
			return m_belief_probabilities[level];
		}	*/

		std::array<std::array<double,noi>, noT> get_irr_beliefs() const
		{
			return m_irr_belief;
		}

		/*std::array<double, noi> get_irr_probabilities(int level) const
		{
			return m_irr_probabilities[level];
		}

		int get_next_free_child() const
		{
			return m_free_child;
		}

		void set_free_child(int counter)
		{
			m_free_counter = counter;
		}

		int get_free_counter() const
		{
			return m_free_counter;
		}*/

		void set_child(int action, True_node* child)
		{
			assert(!m_children[action]);

			m_children[action] = child;

		   /* while(m_free_child < noa && m_action_count[m_free_child])
			{
				++m_free_child;
				//m_free_counter = m_free_child;
			}

			while(m_free_counter < noa && m_action_count[m_free_counter] )
			{
				++m_free_counter;
			}
			/*if(m_free_child < noa)
			{
				m_free_counter = m_free_child;
			}*/
		}

		double get_payoff(int action) const
		{
			return m_payoff[action];
		}

		void set_payoff(int action, double value)
		{
			m_payoff[action] = value;
		}

		std::array<std::array<double, noi>, noT+1> get_shifts() const
		{
			return m_shifts;
		}

		void set_shift(int level, double value, int irr)
		{
			m_shifts[level][irr]=value;
		}

		/*void inc_action_count(int action)
		{
			++m_action_count[action];

		    while(m_free_child < noa && m_action_count[m_free_child])
			{
				++m_free_child;
				//m_free_counter = m_free_child;
			}
			//++m_free_counter;



			while(m_free_counter < noa && m_action_count[m_free_counter] )
			{
				++m_free_counter;
			}

			//else
			//{
			//	++m_free_counter;
				//m_free_counter = m_free_child;
			//}
		}*/

		/*int get_action_count(int action) const
		{
			return m_action_count[action];
		}

		void set_action_count(int action, int counter)
		{
			m_action_count[action] = counter;
		}

		void rollout_done(int guilt, int level)
		{
			m_rolled_out[level][guilt] = true;
		}

		void rollout_undone(int guilt, int level)
		{
			m_rolled_out[level][guilt] = false;
		}*/

		void expectation_undone(int level)
		{
			m_expectation_exists[level] = false;
		}

		bool expectation_set(int level) const
		{
			return m_expectation_exists[level];
		}

		/*bool is_rolled_out(int guilt, int level) const
		{
			return m_rolled_out[level][guilt];
		}*/

		void set_expectation(double value, int level)
		{
			m_expectation[level]= value;
			m_expectation_exists[level] = true;
		}

		void confirm_exploration(int guilt, int level)
		{
			m_sufficient_exploration[level][guilt] = true;
		}

		bool get_confirm_exploration(int guilt, int level) const
		{
			return m_sufficient_exploration[level][guilt];
		}

		double get_expectation(int level) const
		{
			return m_expectation[level];
		}
		/*
		int get_next_exploration(int current_count, double temperature, int horizon) //expects that all children have been constructed!
		{
			std::array<double,noa> exploration;
			double sum =0.0;
			double max_val = -100.0;
			for(int action = 0; action < noa; ++action)
			{
				max_val = ( 1.0/temperature*(m_payoff[action] +
				exploration_constant*sqrt(static_cast<double>(current_count)/(m_action_count[action]))) > max_val ?
1.0/temperature*(m_payoff[action] + exploration_constant*sqrt(static_cast<double>(current_count)/(m_action_count[action]))):max_val);
				exploration[action] = exp(1.0/temperature*(m_payoff[action] + exploration_constant*sqrt(static_cast<double>(current_count)/(m_action_count[action]))));
				sum += exploration[action];
			}

			for(int action = 0; action < noa; ++action)
			{
				if(1.0/temperature*(m_payoff[action] + exploration_constant*sqrt(static_cast<double>(current_count)/(m_action_count[action])))-max_val > -20.0)
				{
					exploration[action] = exploration[action]/sum;
				}
				else
				{
					exploration[action] = 0.0;
				}
			}

			int choice =softmax(exploration, 50);

			return choice;
		}*/


		~True_node()
		{

			for(int i=0; i <noa; ++i)
			{
				if(!(m_children[i]==NULL))
				{
					delete m_children[i];
				}
			}

		}

	private:
		std::array<double, noa> m_payoff;
		std::array<True_node*,noa> m_children;
		//std::array<int, noa> m_action_count;
		std::array<std::array<std::array<double,noa>,nob>, noT+1>  m_exp_payoffs;
		std::array<std::array<bool, nob>, noT+1> m_sufficient_exploration;
		std::array<std::array<double,nob>, noT> m_belief_parameters;
		std::array<std::array<double,noi>, noT> m_irr_belief;
		//std::array<std::array<double,noi>, noT> m_irr_probabilities;
		std::array<std::array<double, noi>, noT+1> m_shifts;
		//std::array<std::array<double,nob>, noT> m_belief_probabilities;
		//int m_free_child;
		//int m_free_counter;
		std::array<bool, noT+1> m_expectation_exists;
		//std::array<std::array<bool, nob>, noT+1> m_rolled_out;
		std::array<double , noT+1> m_expectation;
};

#endif
