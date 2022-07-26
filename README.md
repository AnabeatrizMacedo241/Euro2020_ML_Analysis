# Euro 2020 API ⚽️
<p> Made for the Women In Sports Data Hackaton
</p>
<h4><a href="#introduction">About the API</a> | <a href="#instruction">How to use</a> | <a href="#reference">Reference</a> | <a href="#functions">Documentation</a> | <a href="#code">Github</a> </h4>

<br />

<h2 id="introduction">About the API </h2>
<p>
	Working with data from Euro 2020.
Get insightful charts and informations from player and team performance during the competition using statsbomb and fbref data.
</p>

<br />

<h2 id="instruction">How to use</h2>

<strong>Libraries needed</strong>

    pip install pandas
    pip install time
    pip install matplolib.pyplot
    pip install seaborn
    pip install numpy
    pip install sklearn
    pip install math
    pip install mplsoccer
    pip install soccerpllots

<br />

<strong>Installing the API</strong>

    pip install Euro2020_API
  
 <br/>   
    
<strong>Importing the API</strong>

    from Euro2020_API import Stats

<br />

<h2 id="functions">Methods</h2>

| Method:                | What the method does:                                                                                                   |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------- |
|`SimilarOffense(player_name)`| Returns a dataset with similar players(Forwards and Midfileders) related to the one you choose by a KMeans model.                                                                |
|`SimlarDefense(player_name)`| Returns a dataset with similar players(Defenders and Midfileders) related to the one you choose by a KMeans model.                                                       |
|`RadarChartOff(player1, player2)`| Returns a Radar chart comparing two offensive players performance and a scatterplot that shows their grade and salary.                                                          |
|`RadarChartDef(player1, player2)`| Returns a Radar chart comparing two defensive players performance and a scatterplot that shows their grade and salary.                                                                         |
|`RadarChartGK(player1, player2)`| Returns a Radar chart comparing two Goalkeepers performance and a scatterplot that shows their grade and salary.    |
|`Scorers()`| Returns a dataframe with the players who had high number of goals.    |
|`Passers()`| Returns a dataframe with the players who had high number of assists and passing percentage.    |
|`get_player(player_name)`| Returns a dataframe with every Fbref info abouth them during the Euro.    |
<p>Parameters:
  <li>player_name= Name of the player you want.</li>
  <li>player1,player2 = Name of the two players you desire to compare.</li>  
	</ul>
</p>

<br />

<h2 id="code">Github Repository</h2>

Repository with the documentation and examples of how to use the package. 

<ul>
	<li>https://github.com/AnabeatrizMacedo241/Euro2020_API</li>
	<li>https://github.com/AnabeatrizMacedo241/women_in_sports_hackathon</li>
</ul>

<br />

<h2 id="reference">Reference</h2>

<ul>
	<li>https://fbref.com</li>
	<li>https://github.com/statsbomb</li>
</ul>

<br />
