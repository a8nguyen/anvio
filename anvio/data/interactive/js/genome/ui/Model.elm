module Model exposing (Model, defaultModel)

import Http exposing (..)
import Set exposing (..)
import SharedTypes exposing (Contig)


type alias Model =
    { contigs : List Contig
    , genomes : Set String
    , error : Maybe String
    , screenCenterAsBasePos : Float
    , contigOffset : Float
    , params :
        { basesPerPixel : Int
        , contigBarHeight : Int
        , gap : Int
        }
    }


defaultModel =
    { contigs = []
    , genomes = Set.empty
    , error = Nothing
    , screenCenterAsBasePos = 0.0
    , contigOffset = 0.0
    , params =
        { basesPerPixel = 10000
        , contigBarHeight = 20
        , gap = 5
        }
    }