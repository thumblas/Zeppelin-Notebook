import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS

  def parseRating1(line : String) : (Int,Int,Double,Int) = {
    //println(x)
    val x = line.split("::")
    val userId = x(0).toInt
    val movieId = x(1).toInt
    val rating = x(2).toDouble
    val timeStamp = x(3).toInt/10
    return (userId,movieId,rating,timeStamp)
  }

  def parseRating1(line : String) : (Int,Int,Double,Int) = {
    //println(x)
    val x = line.split("::")
    val userId = x(0).toInt
    val movieId = x(1).toInt
    val rating = x(2).toDouble
    val timeStamp = x(3).toInt/10
    return (userId,movieId,rating,timeStamp)
  }

val moviesFile = sc.textFile("movies.dat")
    val moviesRDD = moviesFile.map(line => line.split("::"))
   //
	 println("Number of movies",moviesRDD.count())
    //
    val ratingsFile = sc.textFile("ratings.dat")
    val ratingsRDD = ratingsFile.map(line => parseRating1(line))
    
	println("ratings count=",ratingsRDD.count())
    

ratingsRDD.take(5).foreach(println) // always check the RDD
    //
    val numRatings = ratingsRDD.count()
    val numUsers = ratingsRDD.map(r => r._1).distinct().count()
    val numMovies = ratingsRDD.map(r => r._2).distinct().count()

    println("Got %d ratings from %d users on %d movies.".format(numRatings, numUsers, numMovies))
     val trainSet = ratingsRDD.filter(x => (x._4 % 10) < 6).map(x=>parseRating(x))
    val validationSet = ratingsRDD.filter(x => (x._4 % 10) >= 6 &  (x._4 % 10) < 8).map(x=>parseRating(x))
    val testSet = ratingsRDD.filter(x => (x._4 % 10) >= 8).map(x=>parseRating(x))
    println("Training: "+ "%d".format(trainSet.count()) + ", validation: " + "%d".format(validationSet.count()) +  ", test: " + "%d".format(testSet.count()) + ".")
    //

val rank = 10
    val numIterations = 20
    val mdlALS = ALS.train(trainSet,rank,numIterations)
    //
    // prepare validation set for prediction
    //
    val userMovie = validationSet.map { 
              case Rating(user, movie, rate) =>(user, movie)
    }
    //
    // Predict and convert to Key-Value PairRDD
    val predictions = mdlALS.predict(userMovie).map {
      case Rating(user, movie, rate) => ((user, movie), rate)
    }
    //
    println(predictions.count())
    predictions.take(5).foreach(println)
    //

val validationPairRDD = validationSet.map(r => ((r.user, r.product), r.rating))
    println(validationPairRDD.count())
    validationPairRDD.take(5).foreach(println)
    println(validationPairRDD.getClass())
    println(predictions.getClass())

val ratingsAndPreds = validationPairRDD.join(predictions) 
    println(ratingsAndPreds.count())
    ratingsAndPreds.take(3).foreach(println)
    //
    val mse = ratingsAndPreds.map(r => {
      math.pow((r._2._1 - r._2._2),2)
    }).reduce(_+_) / ratingsAndPreds.count()

val rmse = math.sqrt(mse)
println("MSE = %2.5f".format(mse) + " RMSE = %2.5f" .format(rmse))
println("Model Created")

moviesFile.toDF.registerTempTable("moviesdata")


%sql
select * from moviesdata

