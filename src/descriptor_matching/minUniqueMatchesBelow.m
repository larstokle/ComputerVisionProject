function matches = minUniqueMatchesBelow(vals, max_val)

% find best match
[minVals, matches] = min(vals,[],1);

% remove too bad matches
matches(minVals > max_val) = 0;

% find unique matches
unique_matches = zeros(size(matches));
[~,unique_match_idxs,~] = unique(matches, 'stable');
unique_matches(unique_match_idxs) = matches(unique_match_idxs);

matches = unique_matches;
end